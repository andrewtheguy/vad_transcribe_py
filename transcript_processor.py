#!/usr/bin/env python3
import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import TextIO, cast

OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:e2b-it-q8_0"
SUMMARY_SYSTEM_PROMPT = "總結一下這節目錄音文本的內容"
STREAM_END_MARKER = '"type": "stream_end"'

BATCH_TRANSCRIPT_BASE = Path("./tmp/transcripts")
BATCH_SUMMARY_BASE = Path("./tmp/summaries")


def format_ts(ms: int) -> str:
    total = ms // 1000
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def convert(path: Path, out: TextIO) -> None:
    last_end_ms = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "transcript":
                continue
            start_ms = int(obj["start_ms"])
            text = obj["text"]
            last_end_ms = int(obj["end_ms"])
            out.write(f"[{format_ts(start_ms)}]:{text}\n")
    out.write(f"[{format_ts(last_end_ms)}]: (end)\n")


def summarize(transcript: str, model: str) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        result = cast("dict[str, object]", json.loads(resp.read()))
    choices = cast("list[dict[str, object]]", result["choices"])
    message = cast("dict[str, object]", choices[0]["message"])
    return cast(str, message["content"])


def summarize_file(src: Path, dest: Path | None, model: str) -> None:
    buf = io.StringIO()
    convert(src, buf)
    summary = summarize(buf.getvalue(), model)
    if dest is None:
        sys.stdout.write(summary)
        if not summary.endswith("\n"):
            sys.stdout.write("\n")
        return
    with dest.open("w", encoding="utf-8") as out:
        out.write(summary)
        if not summary.endswith("\n"):
            out.write("\n")


def has_stream_end(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return False
    if size == 0:
        return False
    with path.open("rb") as f:
        tail_size = min(size, 4096)
        f.seek(size - tail_size)
        tail = f.read(tail_size).decode("utf-8", errors="replace")
    last = next((ln for ln in reversed(tail.splitlines()) if ln.strip()), "")
    return STREAM_END_MARKER in last


def batch_summarize(model: str) -> int:
    if not BATCH_TRANSCRIPT_BASE.is_dir():
        print(
            f"Error: transcript dir not found: {BATCH_TRANSCRIPT_BASE}",
            file=sys.stderr,
        )
        return 1

    suffix = "_transcript.jsonl"
    jsonls = sorted(
        p for p in BATCH_TRANSCRIPT_BASE.rglob(f"*{suffix}") if p.is_file()
    )
    for jsonl in jsonls:
        if not has_stream_end(jsonl):
            print(f"Skipping incomplete: {jsonl}", file=sys.stderr)
            continue

        rel = jsonl.relative_to(BATCH_TRANSCRIPT_BASE)
        base = jsonl.name[: -len(suffix)]
        out_file = BATCH_SUMMARY_BASE / rel.parent / f"{base}_summary.md"

        if out_file.is_file() and out_file.stat().st_size > 0:
            print(f"Already summarized: {out_file}")
            continue

        out_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Summarizing: {jsonl}", flush=True)
        try:
            summarize_file(jsonl, out_file, model)
        except Exception as e:
            print(f"Error: failed to summarize {jsonl}: {e}", file=sys.stderr)
            return 1

    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert, summarize, or batch-summarize transcript jsonl files."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_convert = sub.add_parser(
        "convert", help="Convert transcript jsonl to timestamped text."
    )
    p_convert.add_argument(
        "--input", "-i", type=Path, required=True, help="transcript.jsonl input path"
    )
    p_convert.add_argument(
        "--output", "-o", type=Path, help="output path (default: stdout)"
    )

    p_sum = sub.add_parser(
        "summarize", help="Summarize a single transcript via Ollama."
    )
    p_sum.add_argument(
        "--input", "-i", type=Path, required=True, help="transcript.jsonl input path"
    )
    p_sum.add_argument(
        "--output", "-o", type=Path, help="output path (default: stdout)"
    )
    p_sum.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )

    p_batch = sub.add_parser(
        "batch-summarize",
        help=f"Summarize every *_transcript.jsonl under {BATCH_TRANSCRIPT_BASE}.",
    )
    p_batch.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()
    cmd = cast(str, args.cmd)

    if cmd == "convert":
        src = cast(Path, args.input)
        dest = cast("Path | None", args.output)
        if dest is None:
            convert(src, sys.stdout)
        else:
            with dest.open("w", encoding="utf-8") as out:
                convert(src, out)
        return 0

    if cmd == "summarize":
        src = cast(Path, args.input)
        dest = cast("Path | None", args.output)
        model = cast(str, args.model)
        summarize_file(src, dest, model)
        return 0

    if cmd == "batch-summarize":
        model = cast(str, args.model)
        return batch_summarize(model)

    return 1


if __name__ == "__main__":
    sys.exit(main())
