"""Download and cache Moonshine ONNX model files."""

import logging
import os
import sys
from pathlib import Path

import requests
from filelock import FileLock
from platformdirs import user_cache_dir
from tqdm import tqdm

from .models import ModelArch, STREAMING_ARCHS

logger = logging.getLogger(__name__)

APP_NAME = "moonshine_voice"


def get_cache_dir() -> Path:
    env_var = f"{APP_NAME.upper()}_CACHE"
    return Path(os.environ.get(env_var, user_cache_dir(APP_NAME)))


def _write_stream(response: requests.Response, partial: Path, existing_size: int) -> None:
    """Write a streaming response to a partial file."""
    if response.status_code not in (200, 206):
        response.raise_for_status()

    if response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            total = int(content_range.split("/")[-1])
        else:
            cl = response.headers.get("Content-Length")
            total = existing_size + int(cl) if cl else None
    else:
        existing_size = 0
        partial.unlink(missing_ok=True)
        cl = response.headers.get("Content-Length")
        total = int(cl) if cl else None

    mode = "ab" if existing_size > 0 else "wb"
    with open(partial, mode) as f, tqdm(
        total=total,
        initial=existing_size,
        unit="B",
        unit_scale=True,
        desc=partial.stem,
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def _download_file(url: str, dest: Path, timeout: int = 30) -> Path:
    """Download a file with resume support and atomic writes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    lock = FileLock(str(dest) + ".lock")

    with lock:
        if dest.exists():
            return dest

        existing_size = partial.stat().st_size if partial.exists() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        # Try with Range header first; on 416 retry from scratch
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as response:
            if response.status_code == 416:
                existing_size = 0
                partial.unlink(missing_ok=True)
                # Fall through — retry below
            else:
                _write_stream(response, partial, existing_size)
                partial.rename(dest)
                return dest

        # Retry without Range header
        with requests.get(url, timeout=timeout, stream=True) as response:
            _write_stream(response, partial, 0)

        partial.rename(dest)
    return dest


def _get_components(arch: ModelArch, language: str) -> list[str]:
    """Return the list of files to download for a given architecture."""
    if arch in STREAMING_ARCHS:
        components = [
            "adapter.ort",
            "cross_kv.ort",
            "decoder_kv.ort",
            "encoder.ort",
            "frontend.ort",
            "streaming_config.json",
            "tokenizer.bin",
        ]
        if language == "en":
            components.append("decoder_kv_with_attention.ort")
    else:
        components = [
            "encoder_model.ort",
            "decoder_model_merged.ort",
            "tokenizer.bin",
        ]
        if language == "en":
            components.append("decoder_with_attention.ort")
    return components


def download_model(language: str, arch: ModelArch, download_url: str) -> str:
    """Download model files and return the local model directory path."""
    cache_dir = get_cache_dir()
    model_folder_name = download_url.replace("https://", "")
    root_model_path = cache_dir / model_folder_name
    components = _get_components(arch, language)
    for component in components:
        component_url = f"{download_url}/{component}"
        component_path = root_model_path / component
        _download_file(component_url, component_path)
    return str(root_model_path)
