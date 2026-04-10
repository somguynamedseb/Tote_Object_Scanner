#!/usr/bin/env python3
"""
Clean Point Cloud
==================
Removes noise and small floating clusters from an Nx3 .npy point cloud.

Three filters are applied in order:

  1. Statistical outlier removal
     For each point, look at its K nearest neighbors. Points whose mean
     neighbor distance is more than std_ratio standard deviations above
     the global average get removed. Kills sparse, isolated noise.

  2. Radius outlier removal
     Drops any point with fewer than min_neighbors points inside the
     given radius. Effective against small floating blobs because the
     blob simply doesn't have enough neighbors to pass the threshold.

  3. Small cluster removal (DBSCAN)
     Clusters the cloud and drops any cluster smaller than
     min_cluster_size. The most aggressive option — literally finds the
     floating groups and removes them.

Usage:
    python clean_pointcloud.py pointcloud.npy
    python clean_pointcloud.py pointcloud.npy -o cleaned.npy

All three filters are on by default with sensible values for scans in
millimeters. Tune via CLI flags; disable individual filters with
--no-statistical, --no-radius, --no-cluster.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d is required — install with: pip install open3d", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Clean a stereo point cloud")
    parser.add_argument("npy", help="Input Nx3 .npy point cloud")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .npy (default: cleaned_<input>.npy)")

    # Statistical outlier removal
    parser.add_argument("--no-statistical", dest="statistical",
                        action="store_false",
                        help="Disable statistical outlier removal")
    parser.set_defaults(statistical=False)
    parser.add_argument("--stat-neighbors", type=int, default=20,
                        help="K neighbors for statistical filter (default: 20)")
    parser.add_argument("--stat-std-ratio", type=float, default=2.0,
                        help="Std multiplier for statistical filter (default: 2.0)")

    # Radius outlier removal
    parser.add_argument("--no-radius", dest="radius_filter",
                        action="store_false",
                        help="Disable radius outlier removal")
    parser.set_defaults(radius_filter=False)
    parser.add_argument("--radius", type=float, default=3.0,
                        help="Radius in mm for radius filter (default: 3.0)")
    parser.add_argument("--min-neighbors", type=int, default=16,
                        help="Min neighbors within radius (default: 16)")

    # DBSCAN small-cluster removal
    parser.add_argument("--no-cluster", dest="cluster_filter",
                        action="store_false",
                        help="Disable DBSCAN small-cluster removal")
    parser.set_defaults(cluster_filter=True)
    parser.add_argument("--eps", type=float, default=5.0,
                        help="DBSCAN epsilon in mm (default: 5.0)")
    parser.add_argument("--min-cluster-size", type=int, default=200,
                        help="Drop clusters smaller than this (default: 200)")
    parser.add_argument("--dbscan-min-points", type=int, default=10,
                        help="DBSCAN core-point threshold (default: 10)")

    args = parser.parse_args()

    in_path = Path(args.npy)
    if not in_path.exists():
        log.error("Not found: %s", in_path)
        sys.exit(1)

    pts = np.load(in_path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        log.error("Expected Nx3 array, got %s", pts.shape)
        sys.exit(1)

    n_start = len(pts)
    log.info("Loaded %d points from %s", n_start, in_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    # ── 1. Statistical outlier removal ──
    if args.statistical:
        before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.stat_neighbors,
            std_ratio=args.stat_std_ratio,
        )
        after = len(pcd.points)
        log.info("Statistical filter: %d → %d  (removed %d, %.1f%%)",
                 before, after, before - after,
                 100 * (before - after) / max(before, 1))

    # ── 2. Radius outlier removal ──
    if args.radius_filter:
        before = len(pcd.points)
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=args.min_neighbors,
            radius=args.radius,
        )
        after = len(pcd.points)
        log.info("Radius filter:      %d → %d  (removed %d, %.1f%%)",
                 before, after, before - after,
                 100 * (before - after) / max(before, 1))

    # ── 3. DBSCAN small-cluster removal ──
    if args.cluster_filter and len(pcd.points) > 0:
        before = len(pcd.points)
        labels = np.asarray(pcd.cluster_dbscan(
            eps=args.eps,
            min_points=args.dbscan_min_points,
            print_progress=False,
        ))

        # label == -1 is noise; positive labels are clusters
        if (labels >= 0).any():
            # Count sizes of each cluster
            cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
            big_clusters = cluster_ids[counts >= args.min_cluster_size]
            keep_mask = np.isin(labels, big_clusters)

            n_clusters = len(cluster_ids)
            n_kept = len(big_clusters)
            log.info("DBSCAN found %d clusters; keeping %d with ≥%d points",
                     n_clusters, n_kept, args.min_cluster_size)

            all_points = np.asarray(pcd.points)
            kept = all_points[keep_mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(kept)
        else:
            log.warning("DBSCAN found no clusters; skipping cluster filter")

        after = len(pcd.points)
        log.info("Cluster filter:     %d → %d  (removed %d, %.1f%%)",
                 before, after, before - after,
                 100 * (before - after) / max(before, 1))

    # ── Save ──
    cleaned = np.asarray(pcd.points, dtype=np.float32)
    n_end = len(cleaned)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.parent / f"cleaned_{in_path.name}"

    np.save(out_path, cleaned)
    log.info("Saved %d points (%.1f%% of original) → %s",
             n_end, 100 * n_end / max(n_start, 1), out_path)


if __name__ == "__main__":
    main()
