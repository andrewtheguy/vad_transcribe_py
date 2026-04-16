#!/usr/bin/env bash
set -euo pipefail

HOURLY_DIR="/Volumes/dasdata/capture881903/output/hourly"
OUTPUT_BASE="./tmp/transcripts"

find "$HOURLY_DIR" -name '*.m4a' -type f | sort | while IFS= read -r m4a; do
    rel="${m4a#$HOURLY_DIR/}"
    dir="$(dirname "$rel")"
    base="$(basename "$rel" .m4a)"

    out_dir="$OUTPUT_BASE/$dir"
    out_file="$out_dir/${base}_transcript.jsonl"

    mkdir -p "$out_dir"
    echo "Transcribing: $m4a"

    if ! ffmpeg -nostdin -hide_banner -loglevel error -i "$m4a" -ac 1 -ar 16000 -f wav -acodec pcm_f32le - \
        | uv run vad-transcribe-py transcribe --stdin --backend qwen-asr-rs --chinese-conversion traditional \
        > "$out_file"; then
        echo "WARN: failed to transcribe $m4a, skipping" >&2
        # rm -f "$out_file"
        continue
    fi
done

echo "Done."
