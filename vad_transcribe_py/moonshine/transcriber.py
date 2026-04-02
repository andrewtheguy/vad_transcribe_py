"""Transcriber: runs Moonshine ONNX models on audio segments via onnxruntime."""

import json
import logging
import os
import re
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from .models import ModelArch
from .tokenizer import decode_tokens, load_tokenizer

logger = logging.getLogger(__name__)

# Matches a space between two CJK characters or CJK punctuation
_CJK = (
    r"[\u4e00-\u9fff"  # CJK Unified Ideographs
    r"\u3400-\u4dbf"  # CJK Extension A
    r"\uf900-\ufaff"  # CJK Compatibility Ideographs
    r"\u3000-\u303f"  # CJK Symbols and Punctuation
    r"\uff00-\uffef]"  # Fullwidth Forms
)
_RE_CJK_SPACE = re.compile(f"(?<={_CJK})\\s+(?={_CJK})")

BOS_TOKEN = 1
EOS_TOKEN = 2


def _run_session(session: ort.InferenceSession, feeds: dict[str, Any]) -> list[np.ndarray]:
    """Run an ONNX inference session and return results as numpy arrays."""
    results: list[np.ndarray] = session.run(None, feeds)  # pyright: ignore[reportAssignmentType]
    return results


def _make_session(path: str, num_threads: int | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if num_threads is not None:
        opts.intra_op_num_threads = num_threads
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [
        provider
        for provider in preferred
        if provider in ort.get_available_providers()
    ]
    if providers:
        return ort.InferenceSession(path, sess_options=opts, providers=providers)
    return ort.InferenceSession(path, sess_options=opts)


# ---------------------------------------------------------------------------
# Non-streaming transcriber (encoder_model.ort + decoder_model_merged.ort)
# ---------------------------------------------------------------------------


class _NonStreamingEngine:
    """Loads encoder + merged decoder ONNX and runs auto-regressive decoding."""

    def __init__(self, model_dir: str, token_limit_factor: float, num_threads: int | None = None):
        self.token_limit_factor = token_limit_factor
        self.vocab = load_tokenizer(os.path.join(model_dir, "tokenizer.bin"))

        enc_path = os.path.join(model_dir, "encoder_model.ort")
        dec_path = os.path.join(model_dir, "decoder_model_merged.ort")
        self.encoder = _make_session(enc_path, num_threads)
        self.decoder = _make_session(dec_path, num_threads)

        # Discover decoder architecture from input metadata
        dec_inputs = {inp.name: inp for inp in self.decoder.get_inputs()}
        self._dec_input_names = set(dec_inputs.keys())

        enc_input_names = [inp.name for inp in self.encoder.get_inputs()]
        self._enc_input_name = enc_input_names[0]

        # Detect KV cache structure from decoder inputs
        kv_inputs = sorted(
            n for n in self._dec_input_names if n.startswith("past_key_values.")
        )

        layer_indices = set()
        for name in kv_inputs:
            parts = name.split(".")
            if len(parts) >= 2:
                try:
                    layer_indices.add(int(parts[1]))
                except ValueError:
                    pass
        self.num_layers = max(layer_indices) + 1 if layer_indices else 0

        self.num_kv_heads = 8
        self.head_dim = 64
        if kv_inputs:
            sample_kv = dec_inputs[kv_inputs[0]]
            shape = sample_kv.shape
            if len(shape) >= 4:
                if isinstance(shape[1], int):
                    self.num_kv_heads = shape[1]
                if isinstance(shape[3], int):
                    self.head_dim = shape[3]

        self._enc_hs_name = "encoder_hidden_states"
        if self._enc_hs_name not in self._dec_input_names:
            for name in self._dec_input_names:
                if "encoder" in name and "past" not in name and "mask" not in name:
                    self._enc_hs_name = name
                    break

        self._has_use_cache = "use_cache_branch" in self._dec_input_names
        self._has_enc_mask = "encoder_attention_mask" in self._dec_input_names

        self._decoder_kv_output_map: dict[str, int] = {}
        self._encoder_kv_output_map: dict[str, int] = {}
        for idx, oname in enumerate(
            o.name for o in self.decoder.get_outputs()
        ):
            if idx == 0:
                continue
            past_name = oname.replace("present.", "past_key_values.")
            if ".decoder." in past_name:
                self._decoder_kv_output_map[past_name] = idx
            elif ".encoder." in past_name:
                self._encoder_kv_output_map[past_name] = idx

        logger.info(
            "Non-streaming engine: %d layers, %d heads, head_dim=%d",
            self.num_layers, self.num_kv_heads, self.head_dim,
        )

    def transcribe(self, audio: npt.NDArray[np.float32]) -> str:
        """Transcribe an audio array (1-D float32, 16kHz) to text."""
        audio_input = audio[np.newaxis, :].astype(np.float32)

        enc_feeds: dict[str, Any] = {self._enc_input_name: audio_input}
        enc_input_names = {inp.name for inp in self.encoder.get_inputs()}
        if "attention_mask" in enc_input_names:
            enc_feeds["attention_mask"] = np.ones(
                audio_input.shape, dtype=np.int64
            )
        enc_out = _run_session(self.encoder, enc_feeds)
        hidden_states = enc_out[0]

        enc_frames = hidden_states.shape[1]
        max_tokens = max(2, int(len(audio) * self.token_limit_factor))

        decoder_kv_cache = {}
        encoder_kv_cache = {}
        for layer in range(self.num_layers):
            for kv_type in ("key", "value"):
                decoder_name = f"past_key_values.{layer}.decoder.{kv_type}"
                if decoder_name in self._dec_input_names:
                    decoder_kv_cache[decoder_name] = np.zeros(
                        (1, self.num_kv_heads, 0, self.head_dim),
                        dtype=np.float32,
                    )

                encoder_name = f"past_key_values.{layer}.encoder.{kv_type}"
                if encoder_name in self._dec_input_names:
                    encoder_kv_cache[encoder_name] = np.zeros(
                        (1, self.num_kv_heads, enc_frames, self.head_dim),
                        dtype=np.float32,
                    )

        tokens: list[int] = []
        input_ids = np.array([[BOS_TOKEN]], dtype=np.int64)

        for step in range(max_tokens):
            feeds = {
                "input_ids": input_ids,
                self._enc_hs_name: hidden_states,
            }
            feeds.update(decoder_kv_cache)
            feeds.update(encoder_kv_cache)

            if self._has_use_cache:
                feeds["use_cache_branch"] = np.array([step > 0], dtype=bool)
            if self._has_enc_mask:
                feeds["encoder_attention_mask"] = np.ones(
                    (1, enc_frames), dtype=np.int64
                )

            dec_out = _run_session(self.decoder, feeds)
            logits = dec_out[0]
            next_token = int(np.argmax(logits[0, -1]))

            if next_token == EOS_TOKEN:
                break

            tokens.append(next_token)
            input_ids = np.array([[next_token]], dtype=np.int64)

            for past_name, out_idx in self._decoder_kv_output_map.items():
                if past_name in decoder_kv_cache:
                    decoder_kv_cache[past_name] = dec_out[out_idx]

            if step == 0:
                for past_name, out_idx in self._encoder_kv_output_map.items():
                    if past_name in encoder_kv_cache:
                        encoder_kv_cache[past_name] = dec_out[out_idx]

        return decode_tokens(tokens, self.vocab)


# ---------------------------------------------------------------------------
# Streaming transcriber (frontend + encoder + adapter + cross_kv + decoder_kv)
# ---------------------------------------------------------------------------


class _StreamingEngine:
    """Loads streaming ONNX models and runs the full pipeline on a segment."""

    def __init__(self, model_dir: str, token_limit_factor: float, num_threads: int | None = None):
        self.token_limit_factor = token_limit_factor
        self.vocab = load_tokenizer(os.path.join(model_dir, "tokenizer.bin"))

        config_path = os.path.join(model_dir, "streaming_config.json")
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.frontend = _make_session(os.path.join(model_dir, "frontend.ort"), num_threads)
        self.encoder = _make_session(os.path.join(model_dir, "encoder.ort"), num_threads)
        self.adapter = _make_session(os.path.join(model_dir, "adapter.ort"), num_threads)
        self.cross_kv = _make_session(os.path.join(model_dir, "cross_kv.ort"), num_threads)
        self.decoder_kv = _make_session(os.path.join(model_dir, "decoder_kv.ort"), num_threads)

        self.depth = self.cfg["depth"]
        self.nheads = self.cfg["nheads"]
        self.head_dim = self.cfg["head_dim"]
        self.encoder_dim = self.cfg["encoder_dim"]
        self.decoder_dim = self.cfg.get("decoder_dim", self.encoder_dim)
        self.d_model_frontend = self.cfg.get("d_model_frontend", self.encoder_dim)
        self.c1 = self.cfg.get("c1", self.d_model_frontend * 2)
        self.c2 = self.cfg.get("c2", self.d_model_frontend)
        self.frame_len = self.cfg.get("frame_len", 80)
        self.total_lookahead = self.cfg.get("total_lookahead", 16)
        self.max_seq_len = self.cfg.get("max_seq_len", 448)

        self._frontend_input_names = [
            inp.name for inp in self.frontend.get_inputs()
        ]

        dec_input_names = [inp.name for inp in self.decoder_kv.get_inputs()]
        self._dec_input_map: dict[str, str] = {}
        for name in dec_input_names:
            nl = name.lower()
            if "token" in nl or "input_id" in nl:
                self._dec_input_map["token"] = name
            elif "k_self" in nl and "out" not in nl:
                self._dec_input_map["k_self"] = name
            elif "v_self" in nl and "out" not in nl:
                self._dec_input_map["v_self"] = name
            elif "k_cross" in nl:
                self._dec_input_map["k_cross"] = name
            elif "v_cross" in nl:
                self._dec_input_map["v_cross"] = name

        dec_output_names = [o.name for o in self.decoder_kv.get_outputs()]
        self._dec_out_map: dict[str, int] = {}
        for i, name in enumerate(dec_output_names):
            nl = name.lower()
            if "logit" in nl:
                self._dec_out_map["logits"] = i
            elif "k_self" in nl or (
                "out" in nl and "k" in nl and "cross" not in nl
            ):
                self._dec_out_map["k_self"] = i
            elif "v_self" in nl or (
                "out" in nl and "v" in nl and "cross" not in nl
            ):
                self._dec_out_map["v_self"] = i

        logger.info(
            "Streaming engine: depth=%d, nheads=%d, head_dim=%d, encoder_dim=%d",
            self.depth, self.nheads, self.head_dim, self.encoder_dim,
        )

    def transcribe(self, audio: npt.NDArray[np.float32]) -> str:
        """Transcribe a full audio segment through the streaming pipeline."""
        features = self._run_frontend(audio)
        if features.shape[0] == 0:
            return ""

        encoded = self._run_encoder(features)
        if encoded.shape[0] == 0:
            return ""

        memory = self._run_adapter(encoded)
        k_cross, v_cross = self._run_cross_kv(memory)

        max_tokens = max(2, int(len(audio) * self.token_limit_factor))
        max_tokens = min(max_tokens, self.max_seq_len)
        tokens = self._decode(k_cross, v_cross, max_tokens)

        return decode_tokens(tokens, self.vocab)

    def _run_frontend(self, audio: npt.NDArray[np.float32]) -> np.ndarray:
        sample_buffer = np.zeros((1, 79), dtype=np.float32)
        sample_len = np.array([0], dtype=np.int64)
        conv1_buffer = np.zeros(
            (1, self.d_model_frontend, 4), dtype=np.float32
        )
        conv2_buffer = np.zeros((1, self.c1, 4), dtype=np.float32)
        frame_count = np.array([0], dtype=np.int64)

        audio_chunk = audio[np.newaxis, :].astype(np.float32)

        feeds = {}
        for name in self._frontend_input_names:
            nl = name.lower()
            if "audio" in nl:
                feeds[name] = audio_chunk
            elif "sample_buffer" in nl and "len" not in nl and "out" not in nl:
                feeds[name] = sample_buffer
            elif "sample_len" in nl and "out" not in nl:
                feeds[name] = sample_len
            elif "conv1" in nl and "out" not in nl:
                feeds[name] = conv1_buffer
            elif "conv2" in nl and "out" not in nl:
                feeds[name] = conv2_buffer
            elif "frame_count" in nl and "out" not in nl:
                feeds[name] = frame_count

        out = _run_session(self.frontend, feeds)
        features = out[0]
        return features[0]

    def _run_encoder(self, features: np.ndarray) -> np.ndarray:
        enc_input = features[np.newaxis, :]
        enc_input_name = self.encoder.get_inputs()[0].name
        out = _run_session(self.encoder, {enc_input_name: enc_input})
        encoded = out[0]
        return encoded[0]

    def _run_adapter(self, encoded: np.ndarray) -> np.ndarray:
        adapter_inputs: dict[str, Any] = {}

        for inp in self.adapter.get_inputs():
            name = inp.name
            nl = name.lower()
            if "encoded" in nl or "input" in nl:
                adapter_inputs[name] = encoded[np.newaxis, :]
            elif "pos" in nl or "offset" in nl:
                adapter_inputs[name] = np.array([0], dtype=np.int64)

        out = _run_session(self.adapter, adapter_inputs)
        memory = out[0]
        return memory[0]

    def _run_cross_kv(
        self, memory: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mem_input = memory[np.newaxis, :]
        input_name = self.cross_kv.get_inputs()[0].name
        out = _run_session(self.cross_kv, {input_name: mem_input})
        k_cross = out[0]
        v_cross = out[1]
        return k_cross, v_cross

    def _decode(
        self,
        k_cross: np.ndarray,
        v_cross: np.ndarray,
        max_tokens: int,
    ) -> list[int]:
        k_self = np.zeros(
            (self.depth, 1, self.nheads, 0, self.head_dim), dtype=np.float32
        )
        v_self = np.zeros(
            (self.depth, 1, self.nheads, 0, self.head_dim), dtype=np.float32
        )

        im = self._dec_input_map
        om = self._dec_out_map

        tokens: list[int] = []
        current_token = BOS_TOKEN

        for _ in range(max_tokens):
            feeds = {
                im["token"]: np.array([[current_token]], dtype=np.int64),
                im["k_self"]: k_self,
                im["v_self"]: v_self,
                im["k_cross"]: k_cross,
                im["v_cross"]: v_cross,
            }

            dec_out = _run_session(self.decoder_kv, feeds)

            logits = dec_out[om["logits"]]
            next_token = int(np.argmax(logits[0, -1]))

            if next_token == EOS_TOKEN:
                break

            tokens.append(next_token)
            current_token = next_token

            if "k_self" in om:
                k_self = dec_out[om["k_self"]]
            if "v_self" in om:
                v_self = dec_out[om["v_self"]]

        return tokens


# ---------------------------------------------------------------------------
# Public Transcriber class
# ---------------------------------------------------------------------------


class Transcriber:
    def __init__(
        self,
        model_dir: str,
        model_arch: ModelArch,
        is_streaming: bool,
        strip_cjk_spaces: bool,
        token_limit_factor: float,
        num_threads: int | None = None,
    ):
        self.strip_cjk_spaces = strip_cjk_spaces

        if is_streaming:
            self._engine = _StreamingEngine(model_dir, token_limit_factor, num_threads)
        else:
            self._engine = _NonStreamingEngine(model_dir, token_limit_factor, num_threads)

    def transcribe_chunk(self, audio: npt.NDArray[np.float32]) -> str:
        """Transcribe a single audio chunk (float32, 16kHz) to text."""
        text = self._engine.transcribe(audio)
        if self.strip_cjk_spaces:
            text = _RE_CJK_SPACE.sub("", text)
        return text
