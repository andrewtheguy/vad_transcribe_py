#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import TextIO


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


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <transcript.jsonl> [output.txt]")
    src = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        with Path(sys.argv[2]).open("w", encoding="utf-8") as out:
            convert(src, out)
    else:
        convert(src, sys.stdout)


if __name__ == "__main__":
    main()
