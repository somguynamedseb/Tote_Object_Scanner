#!/usr/bin/env python3
"""
Stereo Depth Test Harness
==========================
Runs OpenCV StereoSGBM across multiple parameter presets on stereo image
pairs from gcode_video_capture.py output. Supports two pair modes:

  1. same_pass  — left + right camera at the same gantry position
                  (true ~20 mm physical baseline from factory calibration).

  2. cross_pass — two frames from different passes at the same X, where
                  the baseline is the gantry-Y offset between the two
                  frames plus any per-lens Y offset.

Image filenames encode the gantry position from extract_frames.py:
    frame_0042_x123.456_y75.000.png

Camera geometry (per user setup):
    - Image horizontal axis  → gantry Y
    - Image vertical axis    → gantry X
    - Left  lens (CAM_B): recorded_y + 10 mm on gantry Y
    - Right lens (CAM_C): recorded_y - 10 mm on gantry Y
    - Lenses are 20 mm apart along gantry-Y (factory baseline)

Because every supported pair has its baseline along gantry-Y, and gantry-Y
maps to the horizontal image axis, SGBM can operate directly on the native
images without any rotation. Disparity is horizontal in pixels.

Depth conversion:
    Z_mm = fx * baseline_mm / disparity_px
where fx is the horizontal focal length from the left camera intrinsics.

Usage:
    python depth_test.py pairs.yaml
    python depth_test.py pairs.yaml -o depth_results/

Example pairs.yaml:

    output_dir: depth_results

    pairs:
      - name: pass3_samepos
        mode: same_pass
        left:  frames/pass_003/left/frame_0015_x120.000_y175.000.png
        right: frames/pass_003/right/frame_0015_x120.000_y175.000.png

      - name: pass3_vs_pass5_both_left
        mode: cross_pass
        # The frame whose lens is at the LOWER gantry-Y goes in "right"
        # (it is the one further in the -Y / image-left direction after
        # the horizontal convention is applied). The script auto-orders
        # them if you set "auto_order: true" — otherwise you must supply
        # left/right explicitly.
        auto_order: true
        image_a: frames/pass_003/left/frame_0015_x120.000_y175.000.png
        image_b: frames/pass_005/left/frame_0015_x120.000_y275.000.png
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ============================================================================
# OAK-D SR CALIBRATION — paste output of dump_oak_calibration.py here
# ============================================================================

IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 800
BASELINE_MM  = 20.044639

K_LEFT = np.array([
    [ 798.733337,  0.000000,  659.384094],
    [ 0.000000,  798.472290,  407.747223],
    [ 0.000000,  0.000000,  1.000000],
])

D_LEFT = np.array([ 25.742926, -45.854671,  0.000429,  0.000099,  25.387995,  25.094036, -43.988190,  24.003641,  0.000000,  0.000000,  0.000000,  0.000000,  0.005359, -0.011672])

K_RIGHT = np.array([
    [ 800.273987,  0.000000,  655.359070],
    [ 0.000000,  800.160645,  389.377899],
    [ 0.000000,  0.000000,  1.000000],
])

D_RIGHT = np.array([ 27.734188, -50.279373, -0.000299,  0.001405,  28.442858,  27.038937, -48.279259,  26.962702,  0.000000,  0.000000,  0.000000,  0.000000, -0.006322, -0.004280])

# Rotation: right camera frame → left camera frame
R_RIGHT_TO_LEFT = np.array([
    [ 0.999990, -0.004177, -0.001550],
    [ 0.004236,  0.999169,  0.040533],
    [ 0.001380, -0.040540,  0.999177],
])

# Translation (mm): right camera frame → left camera frame
T_RIGHT_TO_LEFT_MM = np.array([-20.043397,  0.222905,  0.010679])

# ============================================================================

# Physical Y-offset of each lens from the recorded gantry Y position.
LEFT_LENS_Y_OFFSET_MM  = +10.0
RIGHT_LENS_Y_OFFSET_MM = -10.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SGBM parameter presets
# ---------------------------------------------------------------------------

def _sgbm(**kw):
    """Build an SGBM matcher from keyword args."""
    return cv2.StereoSGBM_create(**kw)


def preset_fast():
    bs = 5
    return _sgbm(
        minDisparity=0,
        numDisparities=128,
        blockSize=bs,
        P1=8 * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )


def preset_balanced():
    bs = 7
    return _sgbm(
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


def preset_quality():
    bs = 9
    return _sgbm(
        minDisparity=0,
        numDisparities=256,
        blockSize=bs,
        P1=8 * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=150,
        speckleRange=1,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )


def preset_fine_detail():
    """Small block size, high uniqueness — good for textured surfaces."""
    bs = 3
    return _sgbm(
        minDisparity=0,
        numDisparities=160,
        blockSize=bs,
        P1=8 * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        disp12MaxDiff=1,
        uniquenessRatio=20,
        speckleWindowSize=100,
        speckleRange=1,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def preset_wide_range():
    """Large disparity range for close objects / big baselines."""
    bs = 7
    return _sgbm(
        minDisparity=0,
        numDisparities=384,
        blockSize=bs,
        P1=8 * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        disp12MaxDiff=2,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


PRESETS = {
    "fast":        preset_fast,
    "balanced":    preset_balanced,
    "quality":     preset_quality,
    "fine_detail": preset_fine_detail,
    "wide_range":  preset_wide_range,
}


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r"frame_(\d+)_x([-\d.]+)_y([-\d.]+)\.png$")


def parse_frame_name(path: str) -> dict:
    """Pull frame index and gantry position out of the filename."""
    m = _FNAME_RE.search(str(path))
    if not m:
        raise ValueError(f"Cannot parse frame filename: {path}")
    return {
        "frame": int(m.group(1)),
        "x_mm": float(m.group(2)),
        "y_mm": float(m.group(3)),
    }


def lens_from_path(path: str) -> str:
    """Infer lens side ('left' or 'right') from the parent directory."""
    parent = Path(path).parent.name.lower()
    if parent == "left":
        return "left"
    if parent == "right":
        return "right"
    raise ValueError(f"Cannot infer lens (left/right) from path: {path}")


def lens_y_mm(path: str) -> float:
    """Actual gantry-Y of a lens given the recorded frame position."""
    info = parse_frame_name(path)
    side = lens_from_path(path)
    offset = LEFT_LENS_Y_OFFSET_MM if side == "left" else RIGHT_LENS_Y_OFFSET_MM
    return info["y_mm"] + offset


# ---------------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------------

def rectify_same_pass(img_left: np.ndarray, img_right: np.ndarray):
    """Use factory calibration to rectify a native L/R pair.

    Returns (rect_left, rect_right, fx_rectified, baseline_mm).
    """
    size = (img_left.shape[1], img_left.shape[0])
    T = T_RIGHT_TO_LEFT_MM.reshape(3, 1)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=K_LEFT,
        distCoeffs1=D_LEFT,
        cameraMatrix2=K_RIGHT,
        distCoeffs2=D_RIGHT,
        imageSize=size,
        R=R_RIGHT_TO_LEFT,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        K_LEFT, D_LEFT, R1, P1, size, cv2.CV_16SC2,
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        K_RIGHT, D_RIGHT, R2, P2, size, cv2.CV_16SC2,
    )

    rect_left  = cv2.remap(img_left,  map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, map1_r, map2_r, cv2.INTER_LINEAR)

    fx_rect = float(P1[0, 0])
    baseline_mm = float(np.linalg.norm(T_RIGHT_TO_LEFT_MM))
    return rect_left, rect_right, fx_rect, baseline_mm


def rectify_cross_pass(img_a: np.ndarray, img_b: np.ndarray):
    """Undistort the two frames with the LEFT intrinsics (both are from the
    same camera, or near enough — if one is left and one is right, we still
    undistort each with its own intrinsics below via rectify_cross_pass_ex).

    For the simple same-lens cross-pass case we just undistort with K_LEFT.
    Returns (rect_a, rect_b, fx, principal assumptions).
    """
    size = (img_a.shape[1], img_a.shape[0])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K_LEFT, D_LEFT, size, 0)
    map1, map2 = cv2.initUndistortRectifyMap(
        K_LEFT, D_LEFT, None, new_K, size, cv2.CV_16SC2,
    )
    rect_a = cv2.remap(img_a, map1, map2, cv2.INTER_LINEAR)
    rect_b = cv2.remap(img_b, map1, map2, cv2.INTER_LINEAR)
    fx = float(new_K[0, 0])
    return rect_a, rect_b, fx


# ---------------------------------------------------------------------------
# Disparity → depth → colorization
# ---------------------------------------------------------------------------

def compute_disparity(matcher, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
    """Return float32 disparity in pixels (invalid pixels = NaN)."""
    disp_raw = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp = disp_raw.copy()
    disp[disp <= 0] = np.nan
    return disp


def disparity_to_depth(disp_px: np.ndarray, fx: float, baseline_mm: float) -> np.ndarray:
    """Z_mm = fx * baseline / disparity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = fx * baseline_mm / disp_px
    return depth


def colorize_disparity(disp: np.ndarray) -> np.ndarray:
    """Normalize disparity into 0-255 and apply a colormap for visualization."""
    valid = np.isfinite(disp)
    if not valid.any():
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(disp[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((disp - lo) / (hi - lo), 0, 1)
    norm[~valid] = 0
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored


def colorize_depth(depth_mm: np.ndarray, clip_mm: tuple[float, float] = (50, 2000)) -> np.ndarray:
    lo, hi = clip_mm
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    norm = np.clip((depth_mm - lo) / (hi - lo), 0, 1)
    norm[~valid] = 1.0  # render invalid as far/black
    u8 = ((1.0 - norm) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored


# ---------------------------------------------------------------------------
# Pair handling
# ---------------------------------------------------------------------------

def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def prepare_same_pass(pair: dict):
    left_path  = pair["left"]
    right_path = pair["right"]
    img_l = load_gray(left_path)
    img_r = load_gray(right_path)
    if img_l.shape != img_r.shape:
        raise ValueError(f"Image size mismatch: {left_path} vs {right_path}")

    rect_l, rect_r, fx_rect, baseline = rectify_same_pass(img_l, img_r)
    info = {
        "mode": "same_pass",
        "baseline_mm": baseline,
        "fx": fx_rect,
        "left_path": left_path,
        "right_path": right_path,
    }
    return rect_l, rect_r, info


def prepare_cross_pass(pair: dict):
    """Build a rectified L/R pair from two frames at the same gantry X but
    different gantry Y positions. The image with the SMALLER gantry-Y goes
    on the RIGHT (because SGBM searches from right toward left and the
    "closer to -Y" camera sees content shifted in the same direction as
    a physical right camera would).
    """
    auto = pair.get("auto_order", False)

    if auto:
        a_path = pair["image_a"]
        b_path = pair["image_b"]
        y_a = lens_y_mm(a_path)
        y_b = lens_y_mm(b_path)
        if y_a == y_b:
            raise ValueError(
                "cross_pass pair has identical lens Y positions — "
                "no baseline to compute depth from."
            )
        # Larger gantry-Y → "left" image. Smaller gantry-Y → "right" image.
        if y_a > y_b:
            left_path, right_path = a_path, b_path
        else:
            left_path, right_path = b_path, a_path
    else:
        left_path  = pair["left"]
        right_path = pair["right"]

    img_l = load_gray(left_path)
    img_r = load_gray(right_path)
    if img_l.shape != img_r.shape:
        raise ValueError(f"Image size mismatch: {left_path} vs {right_path}")

    # Sanity check X positions
    info_l = parse_frame_name(left_path)
    info_r = parse_frame_name(right_path)
    if abs(info_l["x_mm"] - info_r["x_mm"]) > 5.0:
        log.warning(
            "cross_pass pair has X mismatch of %.1f mm — results may be poor",
            abs(info_l["x_mm"] - info_r["x_mm"]),
        )

    y_left  = lens_y_mm(left_path)
    y_right = lens_y_mm(right_path)
    baseline_mm = float(y_left - y_right)  # positive by construction

    rect_l, rect_r, fx = rectify_cross_pass(img_l, img_r)

    info = {
        "mode": "cross_pass",
        "baseline_mm": baseline_mm,
        "fx": fx,
        "left_path": left_path,
        "right_path": right_path,
        "y_left_mm": y_left,
        "y_right_mm": y_right,
    }
    return rect_l, rect_r, info


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_pair(pair: dict, out_dir: Path, preset_names: list[str]):
    name = pair["name"]
    mode = pair["mode"]
    log.info("━━━ Pair: %s  (%s) ━━━", name, mode)

    if mode == "same_pass":
        rect_l, rect_r, info = prepare_same_pass(pair)
    elif mode == "cross_pass":
        rect_l, rect_r, info = prepare_cross_pass(pair)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    log.info(
        "  baseline=%.3f mm  fx=%.2f px  image=%dx%d",
        info["baseline_mm"], info["fx"], rect_l.shape[1], rect_l.shape[0],
    )

    pair_dir = out_dir / name
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Save rectified inputs for inspection
    cv2.imwrite(str(pair_dir / "rect_left.png"), rect_l)
    cv2.imwrite(str(pair_dir / "rect_right.png"), rect_r)

    # Side-by-side with horizontal epipolar lines for visual sanity check
    sbs = np.hstack([rect_l, rect_r])
    sbs_color = cv2.cvtColor(sbs, cv2.COLOR_GRAY2BGR)
    for y in range(0, sbs.shape[0], 40):
        cv2.line(sbs_color, (0, y), (sbs.shape[1], y), (0, 255, 0), 1)
    cv2.imwrite(str(pair_dir / "rectified_sidebyside.png"), sbs_color)

    for preset_name in preset_names:
        matcher = PRESETS[preset_name]()
        disp = compute_disparity(matcher, rect_l, rect_r)
        depth = disparity_to_depth(disp, info["fx"], info["baseline_mm"])

        valid_frac = float(np.isfinite(disp).mean())
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        if valid_depth.size:
            d_med = float(np.median(valid_depth))
            d_lo, d_hi = np.percentile(valid_depth, [5, 95])
        else:
            d_med = d_lo = d_hi = float("nan")

        log.info(
            "  %-12s  valid=%5.1f%%  depth median=%7.1f mm  p5..p95=%.0f..%.0f",
            preset_name, valid_frac * 100, d_med, d_lo, d_hi,
        )

        cv2.imwrite(
            str(pair_dir / f"disp_{preset_name}.png"),
            colorize_disparity(disp),
        )
        cv2.imwrite(
            str(pair_dir / f"depth_{preset_name}.png"),
            colorize_depth(depth),
        )
        # Save raw depth as 16-bit PNG in millimeters (clipped to 65535)
        depth_u16 = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_u16 = np.clip(depth_u16, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(pair_dir / f"depth_{preset_name}_mm16.png"), depth_u16)


def main():
    parser = argparse.ArgumentParser(description="SGBM depth-map test harness")
    parser.add_argument("config", help="YAML file listing pairs to test")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--presets", nargs="+", default=None,
                        help=f"Subset of presets to run (default: all). "
                             f"Available: {', '.join(PRESETS)}")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output or cfg.get("output_dir", "depth_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    preset_names = args.presets or list(PRESETS.keys())
    for p in preset_names:
        if p not in PRESETS:
            log.error("Unknown preset: %s", p)
            sys.exit(1)
    log.info("Running presets: %s", ", ".join(preset_names))

    pairs = cfg.get("pairs", [])
    if not pairs:
        log.error("No pairs found in config.")
        sys.exit(1)

    for pair in pairs:
        try:
            process_pair(pair, out_dir, preset_names)
        except Exception as e:
            log.error("Pair '%s' failed: %s", pair.get("name", "?"), e)

    log.info("Done.")


if __name__ == "__main__":
    main()
