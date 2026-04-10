#!/usr/bin/env python3
"""
Depth Map Stitcher & 3D Viewer
================================
Reads a scan session's manifest.json, loads each raw depth .npy,
applies per-frame filtering (depth range, radius crop, SOR),
offsets the points by their grid positions, and produces:
  1. An interactive 3D point cloud viewer (Open3D)
  2. An exported .ply point cloud file

Requirements:
    pip install numpy opencv-python open3d pyyaml

Usage:
    python test_stitching.py path/to/scan_session/
    python test_stitching.py path/to/scan_session/ --max 500 --radius 300
    python test_stitching.py path/to/scan_session/ --sor 20 --sor-std 1.5
    python test_stitching.py path/to/scan_session/ --no-viewer
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
# Intrinsics
# ---------------------------------------------------------------------------

def get_oakd_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    """
    Read intrinsics from a connected OAK-D, scaled to the given resolution.

    Depth is aligned to the LEFT camera (CAM_B) by default in StereoDepth,
    so we read intrinsics from CAM_B and ask the calibration to rescale them
    to match the actual depth map dimensions.
    """
    try:
        import depthai as dai
        with dai.Device() as device:
            calib = device.readCalibration()
            intrinsics = calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_B, width, height,
            )
            fx = intrinsics[0][0]
            fy = intrinsics[1][1]
            cx = intrinsics[0][2]
            cy = intrinsics[1][2]
            log.info("OAK-D intrinsics (for %dx%d): fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                     width, height, fx, fy, cx, cy)
            return fx, fy, cx, cy
    except Exception as e:
        log.warning("Could not read OAK-D calibration (%s). Using scaled defaults.", e)
        default_fx = 450.0 * (width / 640.0)
        default_fy = 450.0 * (height / 400.0)
        default_cx = width / 2.0
        default_cy = height / 2.0
        log.info("Scaled defaults (for %dx%d): fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                 width, height, default_fx, default_fy, default_cx, default_cy)
        return default_fx, default_fy, default_cx, default_cy


# ---------------------------------------------------------------------------
# Depth → filtered point cloud (per frame, matches depth_viewer logic)
# ---------------------------------------------------------------------------

def depth_to_cloud(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    min_depth: float = 0,
    max_depth: float = 10000,
    max_radius: float = 0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    world_x_limits: tuple[float, float] = None,
    world_y_limits: tuple[float, float] = None,
) -> o3d.geometry.PointCloud:
    """
    Back-project a depth frame into a coloured 3D point cloud with filtering.

    Parameters
    ----------
    max_radius : float
        Maximum lateral (XY) distance from the optical axis in mm.
        Points further from center are discarded. 0 = no limit.
    x_offset, y_offset : float
        Gantry grid offset in mm — shifts the cloud so tiles align in world space.
    """
    h, w = depth.shape[:2]

    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float64)
    valid = (z > min_depth) & (z < max_depth)

    z_valid = z[valid]
    x = (uu[valid] - cx) * z_valid / fx
    y = -(vv[valid] - cy) * z_valid / fy

    # Lateral radius filter: keep only points within a cylinder
    if max_radius > 0:
        lateral_dist = np.sqrt(x**2 + y**2)
        within = lateral_dist <= max_radius
        x = x[within]
        y = y[within]
        z_valid = z_valid[within]

    # Apply gantry offset so tiles combine in world coordinates
    x -= x_offset
    y += y_offset 

    if world_x_limits is not None or world_y_limits is not None:
        keep = np.ones(len(x), dtype=bool)
        if world_x_limits is not None:
            keep &= (x <= -world_x_limits[0]) & (x >= -world_x_limits[1])
        if world_y_limits is not None:
            keep &= (y >= world_y_limits[0]) & (y <= world_y_limits[1])
        x = x[keep]
        y = y[keep]
        z_valid = z_valid[keep]

    points = np.column_stack((x, y, z_valid))

    if len(points) == 0:
        return o3d.geometry.PointCloud()

    # Colour by depth
    z_min, z_max = z_valid.min(), z_valid.max()
    z_norm = (z_valid - z_min) / (z_max - z_min + 1e-6)
    colors_bgr = cv2.applyColorMap(
        (z_norm * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET,
    ).reshape(-1, 3)
    colors_rgb = colors_bgr[:, ::-1] / 255.0

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors_rgb)
    return cloud


# ---------------------------------------------------------------------------
# Build combined cloud from all scan frames
# ---------------------------------------------------------------------------

def build_combined_cloud(
    manifest: list[dict],
    scan_dir: Path,
    min_depth: float,
    max_depth: float,
    max_radius: float,
    voxel_size: float,
    sor_neighbors: int,
    sor_std: float,
    fx: float = None,
    fy: float = None,
    cx: float = None,
    cy: float = None,
    world_x_limits: tuple[float, float] = None,
    world_y_limits: tuple[float, float] = None,
) -> o3d.geometry.PointCloud:
    """Load every raw .npy, filter per-frame, then combine into one cloud."""

    intrinsics_resolved = False
    all_clouds: list[o3d.geometry.PointCloud] = []

    for entry in manifest:
        if "raw_npy" not in entry:
            log.warning("No raw_npy for index %d — skipping.", entry["index"])
            continue

        raw_path = Path(entry["raw_npy"])
        if not raw_path.exists():
            raw_path = scan_dir / raw_path.name
        if not raw_path.exists():
            log.warning("File not found: %s — skipping.", raw_path)
            continue

        depth = np.load(str(raw_path))

        # Resolve intrinsics once from the first frame's dimensions
        if not intrinsics_resolved:
            h, w = depth.shape[:2]
            if fx is None or fy is None or cx is None or cy is None:
                _fx, _fy, _cx, _cy = get_oakd_intrinsics(w, h)
            else:
                _fx, _fy, _cx, _cy = fx, fy, cx, cy
            intrinsics_resolved = True

        # Per-frame: back-project + depth range + radius filter
        cloud = depth_to_cloud(
            depth, _fx, _fy, _cx, _cy,
            min_depth=min_depth,
            max_depth=max_depth,
            max_radius=max_radius,
            x_offset=entry["y"],
            y_offset=entry["x"],#i know they are flipped but that fixes orientation issues, will fix later
            world_x_limits=world_x_limits,
            world_y_limits=world_y_limits,
        )

        n_before = len(cloud.points)
        if n_before == 0:
            log.warning("Index %d (X=%.1f Y=%.1f) — 0 points after filtering, skipping.",
                        entry["index"], entry["x"], entry["y"])
            continue

        # Per-frame: statistical outlier removal
        if sor_neighbors > 0:
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=sor_neighbors, std_ratio=sor_std,
            )

        n_after = len(cloud.points)
        log.info("Index %d  (X=%.1f Y=%.1f)  → %d pts  (SOR: %d → %d)",
                 entry["index"], entry["x"], entry["y"],
                 n_before, n_before, n_after)

        all_clouds.append(cloud)

    if not all_clouds:
        log.error("No valid depth data found.")
        sys.exit(1)

    # Merge all per-frame clouds
    combined = all_clouds[0]
    for c in all_clouds[1:]:
        combined += c

    log.info("Combined cloud: %d points from %d frames.", len(combined.points), len(all_clouds))

    # Global voxel downsample on the merged cloud
    if voxel_size > 0:
        combined = combined.voxel_down_sample(voxel_size=voxel_size)
        log.info("After voxel downsample (%.1f mm): %d points.", voxel_size, len(combined.points))

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stitch depth scans into a combined 3D point cloud",
    )
    parser.add_argument("scan_dir",
                        help="Path to scan session folder (manifest.json + .npy files)")
    parser.add_argument("-c", "--config", default="scan_config.yaml",
                        help="Path to the YAML config file (default: scan_config.yaml)")
    parser.add_argument("--min", type=float, default=175,
                        help="Min depth in mm (default: 200)")
    parser.add_argument("--max", type=float, default=475,
                        help="Max depth in mm (default: 450)")
    parser.add_argument("--radius", type=float, default=500,
                        help="Max lateral (XY) distance from optical axis in mm. "
                             "Crops each frame to a cylinder. 0 = no limit. (default: 300)")
    parser.add_argument("--voxel", type=float, default=0,
                        help="Voxel downsample size in mm after merging (0 = off)")
    parser.add_argument("--sor", type=int, default=200,
                        help="Statistical outlier removal: neighbor count per frame (e.g. 20). 0 = off. (400 works well)")
    parser.add_argument("--sor-std", type=float, default=2.0,
                        help="SOR std deviation threshold (default: 2.0)")
    parser.add_argument("--fx", type=float, default=None,
                        help="Camera focal length X (auto from OAK-D if omitted)")
    parser.add_argument("--fy", type=float, default=None,
                        help="Camera focal length Y (auto from OAK-D if omitted)")
    parser.add_argument("--cx", type=float, default=None,
                        help="Camera principal point X (auto from OAK-D if omitted)")
    parser.add_argument("--cy", type=float, default=None,
                        help="Camera principal point Y (auto from OAK-D if omitted)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Skip the interactive 3D viewer, just export .ply")
    parser.add_argument("--world-x-min", type=float, default=50,
                        help="Hard minimum X in world coordinates (mm)")
    parser.add_argument("--world-x-max", type=float, default=400,
                        help="Hard maximum X in world coordinates (mm)")
    parser.add_argument("--world-y-min", type=float, default=75,
                        help="Hard minimum Y in world coordinates (mm)")
    parser.add_argument("--world-y-max", type=float, default=575,
                        help="Hard maximum Y in world coordinates (mm)")
    
    # parser.add_argument("--world-x-min", type=float, default=None,
    #                     help="Hard minimum X in world coordinates (mm)")
    # parser.add_argument("--world-x-max", type=float, default=None,
    #                     help="Hard maximum X in world coordinates (mm)")
    # parser.add_argument("--world-y-min", type=float, default=None,
    #                     help="Hard minimum Y in world coordinates (mm)")
    # parser.add_argument("--world-y-max", type=float, default=None,
    #                     help="Hard maximum Y in world coordinates (mm)")
    
    args = parser.parse_args()
    
    scan_dir = Path(args.scan_dir)
    if not scan_dir.is_dir():
        log.error("Not a directory: %s", scan_dir)
        sys.exit(1)

    manifest = load_manifest(scan_dir)
    if not manifest:
        log.error("Manifest is empty.")
        sys.exit(1)

    log.info("Loaded %d entries from manifest.", len(manifest))
    
    # Build world-coordinate limit tuples (None if not specified)
    world_x_limits = None
    if args.world_x_min is not None or args.world_x_max is not None:
        world_x_limits = (
            args.world_x_min if args.world_x_min is not None else -np.inf,
            args.world_x_max if args.world_x_max is not None else np.inf,
        )
 
    world_y_limits = None
    if args.world_y_min is not None or args.world_y_max is not None:
        world_y_limits = (
            args.world_y_min if args.world_y_min is not None else -np.inf,
            args.world_y_max if args.world_y_max is not None else np.inf,
        )
    # ── Build combined, filtered point cloud ──
    cloud = build_combined_cloud(
        manifest, scan_dir,
        min_depth=args.min,
        max_depth=args.max,
        max_radius=args.radius,
        voxel_size=args.voxel,
        sor_neighbors=args.sor,
        sor_std=args.sor_std,
        world_x_limits=world_x_limits,
        world_y_limits=world_y_limits,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
    )

    # ── Export .ply ──
    ply_path = scan_dir / "combined_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), cloud)
    log.info("Saved point cloud → %s", ply_path)

    # ── Interactive 3D viewer ──
    if not args.no_viewer:
        log.info("Opening 3D viewer — close the window to exit.")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Stitched Depth Scan", width=1280, height=720)
        vis.add_geometry(cloud)

        opt = vis.get_render_option()
        opt.point_size = 1.5
        opt.background_color = np.array([0.1, 0.1, 0.1])

        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat(cloud.get_center())
        ctr.set_zoom(0.5)

        vis.run()
        vis.destroy_window()

    print(f"\nPoint cloud saved → {ply_path}")


if __name__ == "__main__":
    main()