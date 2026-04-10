#!/usr/bin/env python3
"""
Build Frame Index CSV
======================
Scans a capture run directory (the output of extract_frames.py) and writes
a CSV with one column per lens per pass and one row per frame index.

Expected input layout:
    <run_dir>/
        pass_001/
            left/
                frame_0000_x50.000_y75.000.png
                ...
            right/
                frame_0000_x50.000_y75.000.png
                ...
        pass_002/
            ...

Output CSV columns:
    frame, pass_001_left, pass_001_right, pass_002_left, pass_002_right, ...

Missing frames leave the cell blank.

Usage:
    python build_frame_index.py /path/to/frames
    python build_frame_index.py /path/to/frames -o index.csv
    python build_frame_index.py /path/to/frames --absolute
"""

import argparse
import csv
import re
import sys
from pathlib import Path

_FRAME_RE = re.compile(r"frame_(\d+)_")


def frame_index(path: Path) -> int | None:
    m = _FRAME_RE.search(path.name)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Build per-pass frame index CSV.")
    parser.add_argument("run_dir", help="Frames directory (contains pass_NNN subdirs)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: <run_dir>/frame_index.csv)")
    parser.add_argument("--absolute", action="store_true",
                        help="Write absolute paths instead of relative to run_dir")
    parser.add_argument("--skip", type=int, default=1,
                        help="Keep every Nth row (default: 1 = all rows). "
                             "E.g., 3 keeps rows 0, 3, 6, ...")
    parser.add_argument("--lens", choices=["left", "right", "both"], default="both",
                        help="Which lens columns to include (default: both)")
    args = parser.parse_args()

    if args.skip < 1:
        print("--skip must be >= 1", file=sys.stderr)
        sys.exit(1)

    lenses = ("left", "right") if args.lens == "both" else (args.lens,)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    pass_dirs = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("pass_")
    )
    if not pass_dirs:
        print(f"No pass_* subdirectories found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    # table[frame_idx][column_name] = path_str
    table: dict[int, dict[str, str]] = {}
    columns: list[str] = []

    for pass_dir in pass_dirs:
        for lens in lenses:
            lens_dir = pass_dir / lens
            col = f"{pass_dir.name}_{lens}"
            columns.append(col)
            if not lens_dir.is_dir():
                continue
            for png in sorted(lens_dir.glob("frame_*.png")):
                idx = frame_index(png)
                if idx is None:
                    continue
                path_out = png.resolve() if args.absolute else png.relative_to(run_dir)
                table.setdefault(idx, {})[col] = str(path_out)

    if not table:
        print("No frames found.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else run_dir / "frame_index.csv"
    sorted_indices = sorted(table)[::args.skip]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for idx in sorted_indices:
            row = table[idx]
            writer.writerow([row.get(col, "") for col in columns])

    print(f"Wrote {len(sorted_indices)} rows × {len(columns)} cols → {out_path}")


if __name__ == "__main__":
    main()