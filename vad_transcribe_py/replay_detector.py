"""Detect repeated broadcast segments (ads, news replays) in transcription JSONL."""

import json
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher

DEFAULT_WINDOW_SIZE = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_MIN_CHARS = 50


@dataclass
class Segment:
    start: float
    end: float
    start_formatted: str
    end_formatted: str
    text: str


@dataclass
class ReplayMatch:
    """A pair of similar text windows."""
    similarity: float
    a_start: str
    a_end: str
    b_start: str
    b_end: str
    a_text: str
    b_text: str
    a_seg_idx: int
    b_seg_idx: int
    window_size: int


@dataclass
class ReplayGroup:
    """A cluster of replay matches representing the same repeated content."""
    representative_text: str
    occurrences: list[tuple[str, str]] = field(default_factory=list)
    best_similarity: float = 0.0


def load_segments(jsonl_path: str) -> list[Segment]:
    """Load transcription segments from a JSONL file."""
    segments: list[Segment] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "transcription":
                segments.append(Segment(
                    start=obj["start"],
                    end=obj["end"],
                    start_formatted=obj["start_formatted"],
                    end_formatted=obj["end_formatted"],
                    text=obj["text"],
                ))
    return segments


def build_windows(
    segments: list[Segment], window_size: int, min_chars: int
) -> list[tuple[int, str, str, str]]:
    """Build sliding windows of concatenated segment text.

    Returns list of (start_seg_idx, start_formatted, end_formatted, combined_text).
    """
    windows: list[tuple[int, str, str, str]] = []
    for i in range(len(segments) - window_size + 1):
        group = segments[i : i + window_size]
        combined = " ".join(s.text for s in group)
        if len(combined) < min_chars:
            continue
        windows.append((
            i,
            group[0].start_formatted,
            group[-1].end_formatted,
            combined,
        ))
    return windows


def find_replay_matches(
    windows: list[tuple[int, str, str, str]],
    window_size: int,
    threshold: float,
) -> list[ReplayMatch]:
    """Compare all non-overlapping window pairs for similarity."""
    matches: list[ReplayMatch] = []
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            idx_i = windows[i][0]
            idx_j = windows[j][0]
            # Skip overlapping windows
            if abs(idx_i - idx_j) < window_size:
                continue
            sim = SequenceMatcher(None, windows[i][3], windows[j][3]).ratio()
            if sim >= threshold:
                matches.append(ReplayMatch(
                    similarity=sim,
                    a_start=windows[i][1],
                    a_end=windows[i][2],
                    b_start=windows[j][1],
                    b_end=windows[j][2],
                    a_text=windows[i][3],
                    b_text=windows[j][3],
                    a_seg_idx=idx_i,
                    b_seg_idx=idx_j,
                    window_size=window_size,
                ))
    return matches


def cluster_matches(matches: list[ReplayMatch], window_size: int) -> list[ReplayGroup]:
    """Cluster overlapping matches into replay groups.

    Each group represents one piece of content that was broadcast multiple times.
    """
    if not matches:
        return []

    # Sort by first occurrence position, then second
    matches.sort(key=lambda m: (m.a_seg_idx, m.b_seg_idx))

    groups: list[ReplayGroup] = []
    used: set[int] = set()

    for match in matches:
        if id(match) in used:
            continue

        # Collect all occurrences that overlap with this match
        occurrence_indices: set[int] = {match.a_seg_idx, match.b_seg_idx}
        best_sim = match.similarity
        best_text = match.a_text if match.similarity > 0.5 else match.b_text

        # Find other matches that share a segment index (within window overlap)
        for other in matches:
            if id(other) in used:
                continue
            overlaps = False
            for idx in [other.a_seg_idx, other.b_seg_idx]:
                for existing in list(occurrence_indices):
                    if abs(idx - existing) < window_size:
                        overlaps = True
                        break
                if overlaps:
                    break
            if overlaps:
                occurrence_indices.add(other.a_seg_idx)
                occurrence_indices.add(other.b_seg_idx)
                if other.similarity > best_sim:
                    best_sim = other.similarity
                    best_text = other.a_text
                used.add(id(other))

        used.add(id(match))

        # Deduplicate occurrence positions — merge overlapping indices
        sorted_indices = sorted(occurrence_indices)
        merged_ranges: list[list[int]] = []
        for idx in sorted_indices:
            if merged_ranges and idx - merged_ranges[-1][-1] < window_size:
                merged_ranges[-1].append(idx)
            else:
                merged_ranges.append([idx])

        # Only report if there are at least 2 distinct occurrences
        if len(merged_ranges) >= 2:
            group = ReplayGroup(
                representative_text=best_text,
                best_similarity=best_sim,
            )
            for range_indices in merged_ranges:
                # Use min/max segment index for the range
                min_idx = min(range_indices)
                max_idx = max(range_indices)
                group.occurrences.append((
                    f"seg {min_idx}",
                    f"seg {max_idx + window_size - 1}",
                ))
            groups.append(group)

    return groups


def detect_replays(
    jsonl_path: str,
    window_sizes: list[int] | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[ReplayMatch]:
    """Main detection: find replayed broadcast content in a JSONL transcript.

    Args:
        jsonl_path: Path to the JSONL transcript file.
        window_sizes: List of window sizes to try (default: [3, 5]).
        threshold: Minimum similarity ratio (0.0-1.0) to flag as replay.
        min_chars: Minimum combined character count for a window.

    Returns:
        List of ReplayMatch objects, sorted by similarity descending.
    """
    if window_sizes is None:
        window_sizes = [DEFAULT_WINDOW_SIZE]

    segments = load_segments(jsonl_path)
    if not segments:
        return []

    all_matches: list[ReplayMatch] = []
    for ws in window_sizes:
        windows = build_windows(segments, ws, min_chars)
        matches = find_replay_matches(windows, ws, threshold)
        all_matches.extend(matches)

    # Deduplicate: if a smaller window match is fully contained in a larger
    # window match, keep only the larger one (higher confidence)
    all_matches.sort(key=lambda m: m.similarity, reverse=True)
    return all_matches


def format_report(
    matches: list[ReplayMatch],
    max_text_len: int = 120,
) -> str:
    """Format replay matches into a human-readable report."""
    if not matches:
        return "No broadcast replays detected."

    lines: list[str] = []
    lines.append(f"Found {len(matches)} replay match(es):\n")

    # Deduplicate by clustering
    seen_pairs: set[tuple[int, int]] = set()
    unique_matches: list[ReplayMatch] = []
    for m in matches:
        key = (min(m.a_seg_idx, m.b_seg_idx), max(m.a_seg_idx, m.b_seg_idx))
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_matches.append(m)

    for i, m in enumerate(unique_matches, 1):
        preview = m.a_text[:max_text_len]
        if len(m.a_text) > max_text_len:
            preview += "..."
        lines.append(f"  {i}. [{m.similarity:.0%} match]")
        lines.append(f"     Occurrence A: {m.a_start} - {m.a_end}")
        lines.append(f"     Occurrence B: {m.b_start} - {m.b_end}")
        lines.append(f"     Text: {preview}")
        lines.append("")

    return "\n".join(lines)


def detect_replay_cli(args: list[str] | None = None) -> None:
    """CLI entry point for replay detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect repeated broadcast segments in transcription JSONL"
    )
    parser.add_argument("file", help="Path to JSONL transcript file")
    parser.add_argument(
        "--window-size",
        type=int,
        nargs="+",
        default=[DEFAULT_WINDOW_SIZE],
        help=f"Sliding window size(s) in segments (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Similarity threshold 0.0-1.0 (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Minimum characters per window (default: {DEFAULT_MIN_CHARS})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of text report",
    )

    parsed = parser.parse_args(args)

    matches = detect_replays(
        parsed.file,
        window_sizes=parsed.window_size,
        threshold=parsed.threshold,
        min_chars=parsed.min_chars,
    )

    if parsed.json:
        output = []
        for m in matches:
            output.append({
                "similarity": round(m.similarity, 4),
                "a": {"start": m.a_start, "end": m.a_end, "text": m.a_text},
                "b": {"start": m.b_start, "end": m.b_end, "text": m.b_text},
            })
        json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        report = format_report(matches)
        sys.stdout.write(report)


if __name__ == "__main__":
    detect_replay_cli()
