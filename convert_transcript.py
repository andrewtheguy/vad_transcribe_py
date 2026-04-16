#!/usr/bin/env python3
import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import TextIO, cast

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
SUMMARY_SYSTEM_PROMPT = "總結一下這節目錄音文本的內容"


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


def summarize(transcript: str) -> str:
    payload = {
        "model": "local-model",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    }
    req = urllib.request.Request(
        LMSTUDIO_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        result = cast("dict[str, object]", json.loads(resp.read()))
    choices = cast("list[dict[str, object]]", result["choices"])
    message = cast("dict[str, object]", choices[0]["message"])
    return cast(str, message["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert transcript jsonl to timestamped text.")
    _ = parser.add_argument("input", type=Path, help="transcript.jsonl input path")
    _ = parser.add_argument("output", type=Path, nargs="?", help="output path (default: stdout)")
    _ = parser.add_argument(
        "--summarize",
        action="store_true",
        help="send transcript to LM Studio at 127.0.0.1:1234 and emit the summary instead",
    )
    args = parser.parse_args()
    src = cast(Path, args.input)
    dest = cast("Path | None", args.output)
    do_summarize = cast(bool, args.summarize)

    if do_summarize:
        buf = io.StringIO()
        convert(src, buf)
        summary = summarize(buf.getvalue())
        if dest is None:
            sys.stdout.write(summary)
            if not summary.endswith("\n"):
                sys.stdout.write("\n")
        else:
            with dest.open("w", encoding="utf-8") as out:
                _ = out.write(summary)
                if not summary.endswith("\n"):
                    _ = out.write("\n")
        return

    if dest is None:
        convert(src, sys.stdout)
    else:
        with dest.open("w", encoding="utf-8") as out:
            convert(src, out)


if __name__ == "__main__":
    main()
