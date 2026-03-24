#!/usr/bin/env python3
"""
Depth Map Stitcher & 3D Viewer
================================
Reads a scan session's manifest.json and config, loads each raw depth .npy,
offsets the points by their grid positions, and produces:
  1. A combined colormapped depth image (stitched_depth.png)
  2. An interactive 3D point cloud viewer (Open3D)
  3. An exported .ply point cloud file

Requirements:
    pip install numpy opencv-python open3d pyyaml

Usage:
    python stitch_depth.py path/to/scan_session/
    python stitch_depth.py path/to/scan_session/ -c scan_config.yaml
    python stitch_depth.py path/to/scan_session/ --no-viewer   # export only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / Manifest loaders
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(scan_dir: Path) -> list[dict]:
    manifest_path = scan_dir / "manifest.json"
    if not manifest_path.exists():
        log.error("manifest.json not found in %s", scan_dir)
        sys.exit(1)
    with open(manifest_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Depth → point cloud conversion
# ---------------------------------------------------------------------------

def depth_to_points(
    depth: np.ndarray,
    x_offset_mm: float,
    y_offset_mm: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1.0,
    max_depth_mm: float = 10000.0,
    min_depth_mm: float = 100.0,
) -> np.ndarray:
    """
    Convert a 2D depth frame into an Nx3 array of (X, Y, Z) points in mm.

    Uses the camera intrinsics (fx, fy, cx, cy) to back-project each pixel
    into 3D space, then shifts X/Y by the gantry grid offset so that
    captures from different positions tile together in world coordinates.

    Points with depth outside [min_depth_mm, max_depth_mm] are discarded.
    """
    h, w = depth.shape[:2]

    # Pixel coordinate grids
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float64) * depth_scale

    # Mask out invalid / out-of-range depths
    valid = (z > min_depth_mm) & (z < max_depth_mm)

    z_valid = z[valid]

    # Back-project pixels to 3D using pinhole camera model
    x = (uu[valid] - cx) * z_valid / fx + x_offset_mm
    y = (vv[valid] - cy) * z_valid / fy + y_offset_mm

    return np.column_stack((x, y, z_valid))


# ---------------------------------------------------------------------------
# Stitched 2D depth image
# ---------------------------------------------------------------------------

def stitch_depth_images(
    manifest: list[dict],
    scan_dir: Path,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    """
    Place each depth colourmap into a large canvas at the correct grid offset.
    Returns the stitched BGR image.
    """
    h, w = frame_shape

    # Determine the pixel offsets from the manifest x/y (mm).
    # We use the raw mm values directly as pixel offsets for the mosaic;
    # this gives a readable overview even if the scale isn't 1:1.
    xs = [e["x"] for e in manifest]
    ys = [e["y"] for e in manifest]
    x_min, y_min = min(xs), min(ys)

    # Normalise so the first tile starts at (0, 0)
    positions = [(e["x"] - x_min, e["y"] - y_min) for e in manifest]

    canvas_w = int(max(p[0] for p in positions)) + w
    canvas_h = int(max(p[1] for p in positions)) + h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for entry, (px, py) in zip(manifest, positions):
        # Try colourmap PNG first, fall back to generating from raw
        png_key = "depth_png"
        npy_key = "raw_npy"

        if png_key in entry and Path(entry[png_key]).exists():
            tile = cv2.imread(entry[png_key])
        elif npy_key in entry:
            raw_path = entry[npy_key]
            if not Path(raw_path).exists():
                raw_path = str(scan_dir / Path(raw_path).name)
            raw = np.load(raw_path)
            norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            tile = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        else:
            continue

        th, tw = tile.shape[:2]
        ox, oy = int(px), int(py)
        canvas[oy : oy + th, ox : ox + tw] = tile

    return canvas


# ---------------------------------------------------------------------------
# Point cloud building
# ---------------------------------------------------------------------------

def get_oakd_intrinsics() -> tuple[float, float, float, float]:
    """
    Read the depth camera intrinsics from a connected OAK-D device.

    Returns (fx, fy, cx, cy).  If no device is available, falls back to
    typical OAK-D defaults for 640×400 stereo.
    """
    try:
        import depthai as dai
        with dai.Device() as device:
            calib = device.readCalibration()
            # Get intrinsics for the right mono camera (depth reference)
            intrinsics = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_C,
            )
            # intrinsics is a 3×3 matrix: [[fx,0,cx],[0,fy,cy],[0,0,1]]
            fx = intrinsics[0][0]
            fy = intrinsics[1][1]
            cx = intrinsics[0][2]
            cy = intrinsics[1][2]
            log.info(
                "OAK-D intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                fx, fy, cx, cy,
            )
            return fx, fy, cx, cy
    except Exception as e:
        log.warning("Could not read OAK-D calibration (%s). Using defaults.", e)
        # Typical OAK-D defaults for 640×400
        return 450.0, 450.0, 320.0, 200.0


def build_combined_cloud(
    manifest: list[dict],
    scan_dir: Path,
    depth_scale: float,
    max_depth_mm: float,
    min_depth_mm: float,
    fx: float = None,
    fy: float = None,
    cx: float = None,
    cy: float = None,
) -> o3d.geometry.PointCloud:
    """Load every raw .npy, convert to points, combine into one cloud."""

    # Get intrinsics if not provided
    if fx is None or fy is None or cx is None or cy is None:
        fx, fy, cx, cy = get_oakd_intrinsics()

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for entry in manifest:
        npy_key = "raw_npy"
        if npy_key not in entry:
            log.warning("No raw_npy for index %d — skipping.", entry["index"])
            continue

        raw_path = Path(entry[npy_key])
        if not raw_path.exists():
            # Try relative to scan dir
            raw_path = scan_dir / raw_path.name
        if not raw_path.exists():
            log.warning("File not found: %s — skipping.", raw_path)
            continue

        depth = np.load(str(raw_path))
        x_off = entry["x"]
        y_off = entry["y"]

        pts = depth_to_points(
            depth, x_off, y_off,
            fx=fx, fy=fy, cx=cx, cy=cy,
            depth_scale=depth_scale,
            max_depth_mm=max_depth_mm,
            min_depth_mm=min_depth_mm,
        )

        if pts.shape[0] == 0:
            continue

        all_points.append(pts)

        # Colour by depth (Z) — normalise across this frame
        z_vals = pts[:, 2]
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-6)
        colors = cv2.applyColorMap(
            (z_norm * 255).astype(np.uint8).reshape(-1, 1),
            cv2.COLORMAP_JET,
        ).reshape(-1, 3)[:, ::-1] / 255.0  # BGR→RGB, 0-1
        all_colors.append(colors)

        log.info(
            "Loaded index %d  (X=%.1f Y=%.1f)  → %d points",
            entry["index"], x_off, y_off, pts.shape[0],
        )

    if not all_points:
        log.error("No valid depth data found.")
        sys.exit(1)

    combined_pts = np.vstack(all_points)
    combined_clr = np.vstack(all_colors)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(combined_pts)
    cloud.colors = o3d.utility.Vector3dVector(combined_clr)

    log.info("Combined cloud: %d points total.", len(combined_pts))
    return cloud


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stitch depth scans into a combined 3D point cloud",
    )
    parser.add_argument(
        "scan_dir",
        help="Path to a scan session folder (contains manifest.json + .npy files)",
    )
    parser.add_argument(
        "-c", "--config",
        default="scan_config.yaml",
        help="Path to the YAML config file (default: scan_config.yaml)",
    )
    parser.add_argument(
        "--depth-scale", type=float, default=1.0,
        help="Multiplier to convert raw depth values to mm (default: 1.0 = already mm)",
    )
    parser.add_argument(
        "--max-depth", type=float, default=10000.0,
        help="Discard points beyond this depth in mm (default: 10000)",
    )
    parser.add_argument(
        "--min-depth", type=float, default=100.0,
        help="Discard points closer than this depth in mm (default: 100)",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Skip the interactive 3D viewer, just export files",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.0,
        help="Downsample voxel size in mm (0 = no downsampling)",
    )
    parser.add_argument(
        "--fx", type=float, default=None,
        help="Camera focal length X (pixels). Auto-read from OAK-D if omitted.",
    )
    parser.add_argument(
        "--fy", type=float, default=None,
        help="Camera focal length Y (pixels). Auto-read from OAK-D if omitted.",
    )
    parser.add_argument(
        "--cx", type=float, default=None,
        help="Camera principal point X (pixels). Auto-read from OAK-D if omitted.",
    )
    parser.add_argument(
        "--cy", type=float, default=None,
        help="Camera principal point Y (pixels). Auto-read from OAK-D if omitted.",
    )
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    if not scan_dir.is_dir():
        log.error("Not a directory: %s", scan_dir)
        sys.exit(1)

    cfg = load_config(Path(args.config))
    manifest = load_manifest(scan_dir)

    if not manifest:
        log.error("Manifest is empty.")
        sys.exit(1)

    log.info("Loaded %d entries from manifest.", len(manifest))

    # ── Build combined point cloud ──
    cloud = build_combined_cloud(
        manifest, scan_dir,
        depth_scale=args.depth_scale,
        max_depth_mm=args.max_depth,
        min_depth_mm=args.min_depth,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
    )

    if args.voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size=args.voxel_size)
        log.info("After voxel downsampling: %d points.", len(cloud.points))

    # ── Export .ply ──
    ply_path = scan_dir / "combined_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), cloud)
    log.info("Saved point cloud → %s", ply_path)

    # ── Stitched 2D overview ──
    # Grab frame dimensions from the first raw file
    first_npy = None
    for entry in manifest:
        if "raw_npy" in entry:
            p = Path(entry["raw_npy"])
            if not p.exists():
                p = scan_dir / p.name
            if p.exists():
                first_npy = np.load(str(p))
                break

    if first_npy is not None:
        stitched = stitch_depth_images(manifest, scan_dir, first_npy.shape[:2])
        stitched_path = scan_dir / "stitched_depth.png"
        cv2.imwrite(str(stitched_path), stitched)
        log.info("Saved stitched depth image → %s", stitched_path)

    # ── Interactive 3D viewer ──
    if not args.no_viewer:
        log.info("Opening 3D viewer — close the window to exit.")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Depth Scan Point Cloud", width=1280, height=720)
        vis.add_geometry(cloud)

        opt = vis.get_render_option()
        opt.point_size = 1.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

        vis.run()
        vis.destroy_window()

    print(f"\nOutputs saved to: {scan_dir}/")
    print(f"  Point cloud:    combined_cloud.ply")
    if first_npy is not None:
        print(f"  Stitched image: stitched_depth.png")


if __name__ == "__main__":
    main()