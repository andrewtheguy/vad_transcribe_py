# Repetition handling

Whisper-family decoders (and other ASR models that accept a conditioning prompt) sometimes lock onto the prompt and emit it back near-verbatim instead of transcribing the new audio. This document describes how `vad-transcribe-py` detects and recovers from those failures.

## Layers of defense

There are two complementary mechanisms, applied in order:

1. **Forward filter** — before a transcription is used as the conditioning prompt for the *next* segment, `conditioning_context()` (`vad_transcribe_py/_utils.py`) drops it if it is empty, internally repetitive, or a near-duplicate of the prior line. This prevents the model from being primed to repeat itself on the next call.
2. **Per-call retry** — after the model returns, the backend checks whether the output near-duplicates the prior segment. If so, it re-runs the inference without the conditioning prompt and keeps the second result. This recovers segments that have already been corrupted by prompt echo.

Both layers use the same near-duplicate test: `is_near_duplicate()` in `vad_transcribe_py/_utils.py` — case- and whitespace-insensitive `SequenceMatcher` ratio ≥ 0.9.

## Per-call retry behavior

Implemented in three backends:

| Backend | Conditioning arg removed on retry |
|---|---|
| `whisper` | `prompt_ids` |
| `qwen-asr-rs` | `context` |
| `qwen-asr-mlx` | `system_prompt` |

Each `transcribe()` call retries **at most once**. If the retry output is also a duplicate, it is accepted as-is (the audio may genuinely repeat, e.g. a chant).

### Single-text path

Used by `qwen-asr-rs`, `qwen-asr-mlx`, and `whisper` with `--no-sub-timestamps`.

The model returns a single string per VAD segment. If a conditioning prompt was passed in **and** the returned text near-duplicates the prior VAD segment's joined text (`_prior_line`), the call is repeated with the conditioning argument set to `None`.

### Sub-timestamps path (whisper only, default)

Used by `whisper` with `--sub-timestamps` (the default). The HuggingFace pipeline returns multiple chunks per VAD segment with their own start/end timestamps, so the comparison is finer-grained.

The backend walks the chunk list looking for the **first** index `i` where `chunks[i].text` near-duplicates its predecessor:

- For `i == 0`: predecessor is `_prior_last_chunk` — the last chunk emitted by the previous VAD segment. This catches cross-VAD prompt echo.
- For `i ≥ 1`: predecessor is `chunks[i-1].text`. This catches mid-VAD-segment repetition (the model loops within a single call).

When a duplicate is found, the audio is trimmed to start at `chunks[i].timestamp[0]` and re-run with `prompt_ids` dropped. The retried chunks have their timestamps re-based by `+ trim_start_sec` and spliced in: `chunks[:i] + retried_chunks`.

**Retry gating differs by where the duplicate sits:**

- `i == 0` (cross-VAD): retry **only if** an initial prompt was used. With no prompt to remove, retrying with identical audio and args would produce the same result.
- `i ≥ 1` (mid-VAD): **always retry**, even without an initial prompt. The audio trim itself is a meaningful change in input — removing the looping audio gives the model a chance to produce a non-repeating output.

## Clipping single-line repetitions (opt-in)

Even after the per-call retry, a few segments can slip through with degenerate output the model couldn't recover from — a single word looping for the rest of the VAD window (`"… some real speech then dungu dungu dungu dungu …"`) or a single character repeated dozens of times (`"好好好好…"`). These loops carry no information for downstream consumers, but the prefix before the loop often does.

The `--clip-repetitions` flag (off by default) on `vad-transcribe-py transcribe` post-processes each transcript line. When it detects a heavily-repeated pattern, it **truncates from where the run begins**, keeps **one copy of the pattern**, and appends `(indistinguishable speech)`. So `"... some real speech then dungu dungu dungu …"` becomes `"... some real speech then dungu (indistinguishable speech)"` — the meaningful prefix survives, the loop does not.

**Detection** — `clip_repetitive_text()` in `vad_transcribe_py/_utils.py` mirrors the offline `repetition_analyzer` tool's `truncate_hallucinated_repeats`. For each candidate pattern length `pat_len` in `2..max_pattern_len` (default 30), it walks the string looking for `text[i] == text[i - pat_len]` over a continuous run of at least `pat_len * (min_repeats - 1)` chars (default `min_repeats=10`). The first hit wins, and the output is `text[: run_start + pat_len] + "(indistinguishable speech)"`.

Lines shorter than 100 characters are passed through unchanged — short utterances like "yes yes yes" or "好好好" are plausible real speech, not model hallucination.

Timestamps and the `prompt_retry` flag are preserved on the truncated segment. The JSONL record also carries `repetition_clipped: true` when the line was actually shortened by `--clip-repetitions`.

## JSONL output

Every transcript line in the JSONL output carries booleans for recovery handling: `prompt_retry` indicates whether that segment came from the retry path, and `repetition_clipped` indicates whether `--clip-repetitions` changed the text:

```json
{"type": "transcript", "id": "…", "start_ms": 5000, "start_formatted": "00:00:05.000", "text": "…", "end_ms": 8000, "end_formatted": "00:00:08.000", "prompt_retry": true, "repetition_clipped": false}
```

In the sub-timestamps path, only chunks at index `≥ repeat_index` (i.e. chunks produced by the retry call) are flagged `true`; chunks before the trim point keep `prompt_retry: false`.

Downstream consumers can use this flag to surface or audit the cases where the model needed a second attempt.

## Logging

When a retry fires, the backend logs an INFO line:

- Single-text: `Retrying segment without conditioning prompt (near-duplicate of prior)`
- Sub-timestamps: `Retrying from <Xs> (sub-segment repetition)`

These messages are useful for debugging audio that is hitting the retry path frequently.

## Related code

- `vad_transcribe_py/backends/whisper.py` — both retry paths
- `vad_transcribe_py/backends/qwen_rs.py` — single-text retry
- `vad_transcribe_py/backends/qwen_asr_mlx.py` — single-text retry
- `vad_transcribe_py/_utils.py` — `is_near_duplicate`, `is_repetitive`, `conditioning_context`
- `vad_transcribe_py/_types.py` — `TranscribedSegment.prompt_retry`
- `vad_transcribe_py/cli.py` — `write_jsonl_segment` emits the flag
