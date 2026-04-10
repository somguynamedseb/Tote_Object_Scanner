#!/usr/bin/env python3
"""
Build Combined Point Cloud from Stereo Pairs
==============================================
Reads a CSV listing stereo image pairs, runs SGBM (balanced preset) on each
pair, converts disparity to 3D points in the left-camera frame, then
transforms them into gantry/world coordinates using the gantry position
encoded in the filename and the per-lens Y offset.

All points from all pairs are concatenated and written to a single .npy
file (Nx3, float32, millimeters) for later viewing.

CSV format (header required):
    image_a,image_b

Mode is inferred from the paths:
    - If both images are in .../left/ and .../right/ subdirs at the same
      frame index → same_pass
    - Otherwise → cross_pass (auto-ordered by gantry Y)

World coordinate frame:
    X_world = gantry X  (gantry rail axis running along image vertical)
    Y_world = gantry Y  (rail axis running along image horizontal)
    Z_world = height above bed  (+Z = up, camera looks in -Z)

The camera's absolute height is unknown, so Z_world is reported as
-Z_camera (i.e. negative depth). Add a constant offset downstream if you
want absolute bed-height coordinates.

Usage:
    python build_pointcloud.py pairs.csv
    python build_pointcloud.py pairs.csv -o scan.npy
    python build_pointcloud.py pairs.csv --min-depth 100 --max-depth 1500
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# ============================================================================
# OAK-D SR CALIBRATION — paste output of dump_oak_calibration.py here
# ============================================================================

IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 800
BASELINE_MM  = 20.0

K_LEFT = np.array([
    [800.0,   0.0, 640.0],
    [  0.0, 800.0, 400.0],
    [  0.0,   0.0,   1.0],
])

D_LEFT = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

K_RIGHT = np.array([
    [800.0,   0.0, 640.0],
    [  0.0, 800.0, 400.0],
    [  0.0,   0.0,   1.0],
])

D_RIGHT = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

R_RIGHT_TO_LEFT = np.eye(3)
T_RIGHT_TO_LEFT_MM = np.array([-20.0, 0.0, 0.0])

# ============================================================================

LEFT_LENS_Y_OFFSET_MM  = +10.0
RIGHT_LENS_Y_OFFSET_MM = -10.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Balanced SGBM preset (matches depth_test.py)
# ---------------------------------------------------------------------------

def make_sgbm_balanced():
    bs = 7
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=192,
        blockSize=bs,
        P1=8 * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# ---------------------------------------------------------------------------
# Filename / path parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r"frame_(\d+)_x([-\d.]+)_y([-\d.]+)\.png$")


def parse_frame_name(path: str) -> dict:
    m = _FNAME_RE.search(str(path))
    if not m:
        raise ValueError(f"Cannot parse frame filename: {path}")
    return {
        "frame": int(m.group(1)),
        "x_mm": float(m.group(2)),
        "y_mm": float(m.group(3)),
    }


def lens_from_path(path: str) -> str:
    parent = Path(path).parent.name.lower()
    if parent not in ("left", "right"):
        raise ValueError(f"Cannot infer lens from path: {path}")
    return parent


def lens_y_mm(path: str) -> float:
    """Actual gantry-Y position of this frame's lens."""
    info = parse_frame_name(path)
    side = lens_from_path(path)
    off = LEFT_LENS_Y_OFFSET_MM if side == "left" else RIGHT_LENS_Y_OFFSET_MM
    return info["y_mm"] + off


def infer_mode(path_a: str, path_b: str) -> str:
    """same_pass if a/b are left+right at the same (frame, gantry_x, gantry_y);
    cross_pass otherwise."""
    lens_a = lens_from_path(path_a)
    lens_b = lens_from_path(path_b)
    info_a = parse_frame_name(path_a)
    info_b = parse_frame_name(path_b)

    same_pos = (
        info_a["frame"] == info_b["frame"]
        and abs(info_a["x_mm"] - info_b["x_mm"]) < 1e-6
        and abs(info_a["y_mm"] - info_b["y_mm"]) < 1e-6
    )
    if same_pos and {lens_a, lens_b} == {"left", "right"}:
        return "same_pass"
    return "cross_pass"


# ---------------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------------

def rectify_same_pass(img_left, img_right):
    size = (img_left.shape[1], img_left.shape[0])
    T = T_RIGHT_TO_LEFT_MM.reshape(3, 1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_LEFT, D_LEFT, K_RIGHT, D_RIGHT, size,
        R_RIGHT_TO_LEFT, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    m1l, m2l = cv2.initUndistortRectifyMap(K_LEFT, D_LEFT, R1, P1, size, cv2.CV_16SC2)
    m1r, m2r = cv2.initUndistortRectifyMap(K_RIGHT, D_RIGHT, R2, P2, size, cv2.CV_16SC2)
    rect_l = cv2.remap(img_left,  m1l, m2l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_right, m1r, m2r, cv2.INTER_LINEAR)

    fx = float(P1[0, 0])
    fy = float(P1[1, 1])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    baseline = float(np.linalg.norm(T_RIGHT_TO_LEFT_MM))
    return rect_l, rect_r, fx, fy, cx, cy, baseline


def rectify_cross_pass(img_a, img_b):
    size = (img_a.shape[1], img_a.shape[0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K_LEFT, D_LEFT, size, 0)
    m1, m2 = cv2.initUndistortRectifyMap(K_LEFT, D_LEFT, None, new_K, size, cv2.CV_16SC2)
    rect_a = cv2.remap(img_a, m1, m2, cv2.INTER_LINEAR)
    rect_b = cv2.remap(img_b, m1, m2, cv2.INTER_LINEAR)
    fx = float(new_K[0, 0])
    fy = float(new_K[1, 1])
    cx = float(new_K[0, 2])
    cy = float(new_K[1, 2])
    return rect_a, rect_b, fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Pair preparation
# ---------------------------------------------------------------------------

def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def prepare_pair(image_a: str, image_b: str):
    """Rectify the pair and return everything needed to project to 3D in the
    gantry frame."""
    mode = infer_mode(image_a, image_b)

    if mode == "same_pass":
        # Identify which image is left vs right
        lens_a = lens_from_path(image_a)
        if lens_a == "left":
            left_path, right_path = image_a, image_b
        else:
            left_path, right_path = image_b, image_a

        img_l = load_gray(left_path)
        img_r = load_gray(right_path)
        rect_l, rect_r, fx, fy, cx, cy, baseline = rectify_same_pass(img_l, img_r)

        # Anchor the camera frame at the LEFT lens's gantry position
        info = parse_frame_name(left_path)
        anchor_x = info["x_mm"]
        anchor_y = lens_y_mm(left_path)

    elif mode == "cross_pass":
        # Auto-order: the frame at the LARGER gantry-Y is "left"
        y_a = lens_y_mm(image_a)
        y_b = lens_y_mm(image_b)
        if y_a == y_b:
            raise ValueError(
                f"Cross-pass pair has identical lens Y: {image_a} & {image_b}"
            )
        if y_a > y_b:
            left_path, right_path = image_a, image_b
        else:
            left_path, right_path = image_b, image_a

        # Sanity check: X should match
        info_l = parse_frame_name(left_path)
        info_r = parse_frame_name(right_path)
        if abs(info_l["x_mm"] - info_r["x_mm"]) > 5.0:
            log.warning(
                "cross-pass pair X mismatch %.1f mm — epipolar assumption broken",
                abs(info_l["x_mm"] - info_r["x_mm"]),
            )

        img_l = load_gray(left_path)
        img_r = load_gray(right_path)
        rect_l, rect_r, fx, fy, cx, cy = rectify_cross_pass(img_l, img_r)

        baseline = float(lens_y_mm(left_path) - lens_y_mm(right_path))
        anchor_x = info_l["x_mm"]
        anchor_y = lens_y_mm(left_path)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "mode": mode,
        "rect_left": rect_l,
        "rect_right": rect_r,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "baseline_mm": baseline,
        "anchor_x_mm": anchor_x,
        "anchor_y_mm": anchor_y,
        "left_path": left_path,
        "right_path": right_path,
    }


# ---------------------------------------------------------------------------
# Disparity → 3D points → world frame
# ---------------------------------------------------------------------------

def points_from_pair(prep: dict, matcher, min_depth: float, max_depth: float) -> np.ndarray:
    """Compute disparity, project to 3D points in gantry world coordinates."""
    rect_l = prep["rect_left"]
    rect_r = prep["rect_right"]
    fx, fy = prep["fx"], prep["fy"]
    cx, cy = prep["cx"], prep["cy"]
    baseline = prep["baseline_mm"]

    disp = matcher.compute(rect_l, rect_r).astype(np.float32) / 16.0

    valid = disp > 0
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)

    # Depth in mm, in the left-camera frame
    with np.errstate(divide="ignore", invalid="ignore"):
        Z_cam = fx * baseline / disp

    valid &= np.isfinite(Z_cam) & (Z_cam >= min_depth) & (Z_cam <= max_depth)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)

    vs, us = np.nonzero(valid)
    Z = Z_cam[vs, us]
    X_cam = (us - cx) * Z / fx  # +X_cam = image right  = +gantry Y
    Y_cam = (vs - cy) * Z / fy  # +Y_cam = image down   = +gantry X
    # +Z_cam = into scene       = -gantry Z

    # Transform to world / gantry frame, offset by the pair's anchor position
    X_world = prep["anchor_x_mm"] - Y_cam
    Y_world = prep["anchor_y_mm"] - X_cam
    Z_world = -Z

    pts = np.stack([X_world, Y_world, Z_world], axis=1).astype(np.float32)
    return pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_pairs_csv(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "image_a" not in reader.fieldnames \
                or "image_b" not in reader.fieldnames:
            raise ValueError("CSV must have header: image_a,image_b")
        for row in reader:
            a = row["image_a"].strip()
            b = row["image_b"].strip()
            if not a or not b:
                continue
            pairs.append((a, b))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Combine stereo pairs into one point cloud."
    )
    parser.add_argument("csv", help="CSV with image_a,image_b header")
    parser.add_argument("-o", "--output", default="pointcloud.npy",
                        help="Output .npy path (default: pointcloud.npy)")
    parser.add_argument("--min-depth", type=float, default=25.0,
                        help="Discard points closer than this (mm, default: 50)")
    parser.add_argument("--max-depth", type=float, default=500.0,
                        help="Discard points farther than this (mm, default: 500)")
    parser.add_argument("--subsample", type=int, default=50,
                        help="Keep every Nth point (default: 1 = all)")
    parser.add_argument("--base-dir", default=None,
                        help="Resolve relative CSV paths against this directory "
                             "(default: CSV's own directory)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    base_dir = Path(args.base_dir) if args.base_dir else csv_path.parent

    pairs = read_pairs_csv(csv_path)
    if not pairs:
        log.error("No pairs in CSV.")
        sys.exit(1)
    log.info("Loaded %d pairs from %s", len(pairs), csv_path)

    matcher = make_sgbm_balanced()

    all_points: list[np.ndarray] = []
    total_before = 0

    for idx, (a, b) in enumerate(pairs, 1):
        a_path = Path(a)
        b_path = Path(b)
        if not a_path.is_absolute():
            a_path = base_dir / a_path
        if not b_path.is_absolute():
            b_path = base_dir / b_path

        if not a_path.exists():
            log.warning("[%d/%d] missing: %s", idx, len(pairs), a_path)
            continue
        if not b_path.exists():
            log.warning("[%d/%d] missing: %s", idx, len(pairs), b_path)
            continue

        try:
            prep = prepare_pair(str(a_path), str(b_path))
            pts = points_from_pair(prep, matcher, args.min_depth, args.max_depth)
        except Exception as e:
            log.error("[%d/%d] failed: %s", idx, len(pairs), e)
            continue

        total_before += pts.shape[0]
        if args.subsample > 1 and pts.shape[0] > 0:
            pts = pts[::args.subsample]

        log.info(
            "[%d/%d] %s  baseline=%.1f  anchor=(%.1f,%.1f)  pts=%d",
            idx, len(pairs), prep["mode"],
            prep["baseline_mm"], prep["anchor_x_mm"], prep["anchor_y_mm"],
            pts.shape[0],
        )
        all_points.append(pts)

    if not all_points:
        log.error("No points generated.")
        sys.exit(1)

    cloud = np.concatenate(all_points, axis=0)
    log.info("Combined cloud: %d points (before subsample: %d)",
             cloud.shape[0], total_before)
    log.info("Bounds  X: %.1f .. %.1f mm", cloud[:, 0].min(), cloud[:, 0].max())
    log.info("        Y: %.1f .. %.1f mm", cloud[:, 1].min(), cloud[:, 1].max())
    log.info("        Z: %.1f .. %.1f mm", cloud[:, 2].min(), cloud[:, 2].max())

    out_path = Path(args.output)
    np.save(out_path, cloud)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
