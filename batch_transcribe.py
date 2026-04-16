#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

INPUT_DIR = Path("/Volumes/dasdata/capture881903/trimmed")
OUTPUT_BASE = Path("./tmp/transcripts")
STREAM_END_MARKER = '"type": "stream_end"'

FFMPEG_GLOBAL = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
FFMPEG_OUTPUT_OPTS = [
    "-ac",
    "1",
    "-ar",
    "16000",
    "-f",
    "wav",
    "-acodec",
    "pcm_f32le",
]
TRANSCRIBE_CMD = [
    "uv",
    "run",
    "vad-transcribe-py",
    "transcribe",
    "--stdin",
    "--backend",
    "qwen-asr-rs",
    "--chinese-conversion",
    "traditional",
]


def has_stream_end(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return False
    if size == 0:
        return False
    with path.open("rb") as f:
        tail_size = min(size, 4096)
        _ = f.seek(size - tail_size)
        tail = f.read(tail_size).decode("utf-8", errors="replace")
    last = next((ln for ln in reversed(tail.splitlines()) if ln.strip()), "")
    return STREAM_END_MARKER in last


def transcribe(m4a: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Transcribing: {m4a}", flush=True)

    ffmpeg_args = [*FFMPEG_GLOBAL, "-i", str(m4a), *FFMPEG_OUTPUT_OPTS, "-"]
    with out_file.open("wb") as out:
        ffmpeg = subprocess.Popen(ffmpeg_args, stdout=subprocess.PIPE)
        try:
            assert ffmpeg.stdout is not None
            transcribe_proc = subprocess.Popen(
                TRANSCRIBE_CMD, stdin=ffmpeg.stdout, stdout=out
            )
        except BaseException:
            ffmpeg.kill()
            _ = ffmpeg.wait()
            raise
        # Close parent copy so ffmpeg sees SIGPIPE if transcribe exits early.
        ffmpeg.stdout.close()
        try:
            transcribe_rc = transcribe_proc.wait()
            ffmpeg_rc = ffmpeg.wait()
        except BaseException:
            ffmpeg.kill()
            transcribe_proc.kill()
            _ = ffmpeg.wait()
            _ = transcribe_proc.wait()
            raise

    if ffmpeg_rc != 0:
        msg = f"ffmpeg failed for {m4a} (exit {ffmpeg_rc})"
        raise RuntimeError(msg)
    if transcribe_rc != 0:
        msg = f"vad-transcribe-py failed for {m4a} (exit {transcribe_rc})"
        raise RuntimeError(msg)


def main() -> int:
    if not INPUT_DIR.is_dir():
        print(f"Error: input dir not found: {INPUT_DIR}", file=sys.stderr)
        return 1

    m4as = sorted(p for p in INPUT_DIR.rglob("*.m4a") if p.is_file())
    for m4a in m4as:
        rel = m4a.relative_to(INPUT_DIR)
        out_file = OUTPUT_BASE / rel.parent / f"{rel.stem}_transcript.jsonl"
        if has_stream_end(out_file):
            print(f"Already transcribed: {out_file}")
            continue
        try:
            transcribe(m4a, out_file)
        except Exception as e:
            print(f"Error: failed to transcribe {m4a}: {e}", file=sys.stderr)
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
