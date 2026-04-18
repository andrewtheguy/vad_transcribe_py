#!/usr/bin/env python3
import argparse
import io
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple, TextIO, cast

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-e4b"
SUMMARY_SYSTEM_PROMPT = "總結一下這節目錄音文本的內容"
STREAM_END_MARKER = '"type": "stream_end"'

FILENAME_TIME_RANGE_RE = re.compile(r"_(\d{8})_(\d{6})_(\d{6})(?:_|$)")

BATCH_TRANSCRIPT_BASE = Path("./tmp/transcripts")
BATCH_SUMMARY_BASE = Path("./tmp/summaries")


class TimeRange(NamedTuple):
    date: str
    start: str
    end: str


def _to_12h(hms: str) -> str:
    h, m, s = hms.split(":")
    hi = int(h)
    suffix = "am" if hi < 12 else "pm"
    h12 = hi % 12 or 12
    return f"{h12}:{m}:{s}{suffix}"


def _hms_to_seconds(hms: str) -> int:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def format_ts(ms: int, offset_seconds: int = 0) -> str:
    total = (ms // 1000 + offset_seconds) % 86400
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def convert(path: Path, out: TextIO, time_range: "TimeRange | None" = None) -> None:
    offset = _hms_to_seconds(time_range.start) if time_range else 0
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
            out.write(f"[{format_ts(start_ms, offset)}]:{text}\n")
    out.write(f"[{format_ts(last_end_ms, offset)}]: (end)\n")


def extract_time_range(path: Path) -> TimeRange | None:
    m = FILENAME_TIME_RANGE_RE.search(path.stem)
    if not m:
        return None
    date, start, end = m.groups()
    return TimeRange(
        date=f"{date[:4]}-{date[4:6]}-{date[6:8]}",
        start=f"{start[:2]}:{start[2:4]}:{start[4:6]}",
        end=f"{end[:2]}:{end[2:4]}:{end[4:6]}",
    )


def build_system_prompt(time_range: TimeRange | None) -> str:
    if time_range is None:
        return SUMMARY_SYSTEM_PROMPT
    prompt = (
        f"{SUMMARY_SYSTEM_PROMPT}, "
        f"錄音時間範圍: {time_range.date} {_to_12h(time_range.start)} - {_to_12h(time_range.end)}"
    )
    print(f"System prompt: {prompt}", file=sys.stderr)
    return prompt


def summarize(transcript: str, model: str, time_range: TimeRange | None = None) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": build_system_prompt(time_range)},
            {"role": "user", "content": transcript},
        ],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        result = cast("dict[str, object]", json.loads(resp.read()))
    choices = cast("list[dict[str, object]]", result["choices"])
    message = cast("dict[str, object]", choices[0]["message"])
    return cast(str, message["content"])


def summarize_file(src: Path, dest: Path | None, model: str) -> None:
    time_range = extract_time_range(src)
    buf = io.StringIO()
    convert(src, buf, time_range)
    transcript = buf.getvalue()
    # print(
    #     f"--- Transcript to summarize ({src}) ---\n{transcript}"
    #     f"--- End transcript ---",
    #     file=sys.stderr,
    # )
    summary = summarize(transcript, model, time_range)
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
        tr = extract_time_range(src)
        if dest is None:
            convert(src, sys.stdout, tr)
        else:
            with dest.open("w", encoding="utf-8") as out:
                convert(src, out, tr)
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
