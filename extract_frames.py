#!/usr/bin/env python3
"""
Extract Frames from Video Capture Manifest
===========================================
Reads a manifest.json produced by gcode_video_capture.py, extracts every
frame from each video file, and saves them as PNG images.

Each frame's gantry position is computed from:
    x = start_x + (feedrate_mm_min / 60) * (frame_index / fps)
    y = start_y                          (constant per pass)

Output structure:
    <output_dir>/
        pass_001/
            left/
                frame_0000_x50.000_y75.000.png
                ...
            right/
                frame_0000_x50.000_y75.000.png
                ...
        pass_002/
            ...
        frame_log.csv      <- one row per frame with all metadata

Usage:
    python extract_frames.py /path/to/manifest.json
    python extract_frames.py /path/to/manifest.json -o /path/to/output
    python extract_frames.py /path/to/manifest.json --fps 30
    python extract_frames.py /path/to/manifest.json --workers 8
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# PNG compression level 1 (fast) — range is 0-9, where 0 is no compression
# and 9 is maximum. Level 1 gives a good size reduction with minimal CPU cost.
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def _write_frame(filepath: str, frame, params: list) -> None:
    """Write a single frame to disk (called from worker threads)."""
    cv2.imwrite(filepath, frame, params)


def extract_pass_video(
    video_path: str,
    lens: str,
    pass_num: int,
    start_x: float,
    end_x: float,
    start_y: float,
    feedrate: float,
    fps: float,
    out_dir: Path,
    pool: ThreadPoolExecutor,
) -> list[dict]:
    """Extract all frames from one video file using threaded writes.

    Returns a list of dicts, one per frame, with full metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return []

    container_fps = cap.get(cv2.CAP_PROP_FPS)
    if container_fps and container_fps > 0:
        fps = container_fps

    speed_mm_per_sec = feedrate / 60.0
    sec_per_frame = 1.0 / fps

    lens_dir = out_dir / f"pass_{pass_num:03d}" / lens
    lens_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    futures = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx * sec_per_frame
        x = start_x + speed_mm_per_sec * t
        y = start_y

        if x > end_x:
            break

        filename = f"frame_{frame_idx:04d}_x{x:.3f}_y{y:.3f}.png"
        filepath = str(lens_dir / filename)

        # Dispatch the write to the thread pool — cv2.imwrite releases the
        # GIL during the actual I/O and PNG compression, so threads help.
        futures.append(pool.submit(_write_frame, filepath, frame.copy(), PNG_PARAMS))

        rows.append({
            "pass": pass_num,
            "lens": lens,
            "frame": frame_idx,
            "time_s": round(t, 6),
            "x_mm": round(x, 3),
            "y_mm": round(y, 3),
            "file": filepath,
        })

        frame_idx += 1

    cap.release()

    # Wait for all writes for this video to finish
    for fut in as_completed(futures):
        fut.result()  # raises if a write failed

    log.info(
        "  pass %d %s: %d frames extracted (fps=%.1f)",
        pass_num, lens, frame_idx, fps,
    )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract positioned frames from a video-capture manifest.",
    )
    parser.add_argument(
        "manifest",
        help="Path to manifest.json produced by gcode_video_capture.py",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: <manifest_dir>/frames)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS if the AVI container has no FPS metadata (default: 30)",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=8,
        help="Number of threads for parallel frame writes (default: 8)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest_dir = manifest_path.parent

    out_dir = Path(args.output) if args.output else manifest_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    all_rows: list[dict] = []
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for entry in manifest:
            pass_num = entry["pass"]
            start_x = entry["start_x"]
            end_x = entry["end_x"]
            start_y = entry["start_y"]
            feedrate = entry["feedrate"]

            log.info("━━━ Pass %d  (x=%.1f to %.1f, y=%.1f  F%.0f) ━━━",
                     pass_num, start_x, end_x, start_y, feedrate)

            for lens, key in [("left", "left_video"), ("right", "right_video")]:
                video_rel = entry.get(key)
                if video_rel is None:
                    continue

                video_path = Path(video_rel)
                # if not video_path.is_absolute():
                #     video_path = manifest_dir / video_path

                if not video_path.exists():
                    log.warning("Video file not found, skipping: %s", video_path)
                    continue

                rows = extract_pass_video(
                    video_path=str(video_path),
                    lens=lens,
                    pass_num=pass_num,
                    start_x=start_x,
                    end_x=end_x,
                    start_y=start_y,
                    feedrate=feedrate,
                    fps=args.fps,
                    out_dir=out_dir,
                    pool=pool,
                )
                all_rows.extend(rows)

    # Write CSV log
    csv_path = out_dir / "frame_log.csv"
    if all_rows:
        fieldnames = ["pass", "lens", "frame", "time_s", "x_mm", "y_mm", "file"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        log.info("Frame log saved → %s  (%d rows)", csv_path, len(all_rows))

    elapsed = time.monotonic() - t0
    log.info("Done — %d frames extracted to %s in %.1f s", len(all_rows), out_dir, elapsed)


if __name__ == "__main__":
    main()