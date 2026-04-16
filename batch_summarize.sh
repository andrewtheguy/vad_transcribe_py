#!/usr/bin/env bash
set -euo pipefail

TRANSCRIPT_BASE="./tmp/transcripts"
OUTPUT_BASE="./tmp/summaries"

find "$TRANSCRIPT_BASE" -name '*_transcript.jsonl' -type f | sort | while IFS= read -r jsonl; do
    if ! tail -n 1 "$jsonl" | grep -q '"type": "stream_end"'; then
        echo "Skipping incomplete: $jsonl" >&2
        continue
    fi

    rel="${jsonl#$TRANSCRIPT_BASE/}"
    dir="$(dirname "$rel")"
    base="$(basename "$rel" _transcript.jsonl)"

    out_dir="$OUTPUT_BASE/$dir"
    out_file="$out_dir/${base}_summary.txt"

    if [[ -s "$out_file" ]]; then
        echo "Already summarized: $out_file"
        continue
    fi

    mkdir -p "$out_dir"
    echo "Summarizing: $jsonl"

    if ! uv run python convert_transcript.py --summarize "$jsonl" "$out_file"; then
        echo "WARN: failed to summarize $jsonl, skipping" >&2
        rm -f "$out_file"
        continue
    fi
done

echo "Done."
