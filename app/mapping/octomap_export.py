from __future__ import annotations

"""Export dense 2D occupancy (occ2d) and vertical danger bands (z_band2d) from OctoMap.

This is a Phase-2 v0 helper for:
- local debugging (npz dump + matplotlib quick view)
- later ROS2 publishing (OccupancyGrid / PointCloud2)

Examples:
    python -m app.mapping.octomap_export --mask-dir runs/segment/exp2/masks --grid-scale 10 --out-npz runs/debug/map_v0.npz --preview

    # keep pixel tiles export if you want (optional)
    python -m app.mapping.octomap_export --mask-dir runs/segment/exp2/masks --grid-scale 10 --tile-w-px 16 --tile-h-px 16 --min-coverage 0.3 --out-npz runs/debug/map_v0.npz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.mapping.octomap import OctoMap
from app.paths import resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export occ2d + z_band2d from masks via OctoMap")
    p.add_argument("--mask-dir", type=str, default=None, help="Directory containing mask PNG files")
    p.add_argument("--grid-scale", type=int, default=None, help="Pixel-to-grid scale override (default: OctoMap DEFAULT_CONFIG)")

    # Optional: keep pixel-domain tiling (xywh) result in snapshot (not required for v0 export)
    p.add_argument("--tile-w-px", type=int, default=None)
    p.add_argument("--tile-h-px", type=int, default=None)
    p.add_argument("--min-coverage", type=float, default=0.3)

    p.add_argument("--out-npz", type=str, required=True, help="Output .npz path")
    p.add_argument("--preview", action="store_true", help="Show quick matplotlib preview")
    p.add_argument("--use-columns", action="store_true", help="occ2d uses columns (inclusive) instead of blocked_obstacles")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reuse OctoMap CLI defaults for mask dir & grid scale.
    from app.mapping.octomap import get_default_mask_dir  # local import to avoid circular CLI deps
    from app.config import DEFAULT_CONFIG

    mask_dir = resolve_path(args.mask_dir, get_default_mask_dir())
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    grid_scale = int(args.grid_scale) if args.grid_scale is not None else int(DEFAULT_CONFIG.default_grid_scale)

    grid_w, grid_h = OctoMap.infer_grid_size(mask_dir, grid_scale)
    octomap = OctoMap(grid_w=grid_w, grid_h=grid_h, grid_scale=grid_scale)

    from app.mapping.grid_map import load_mask_entries

    mask_list = load_mask_entries(mask_dir, octomap.grid_handler)

    octomap.grid_handler.batch_masks_to_obs(
        mask_list,
        tile_w_px=args.tile_w_px,
        tile_h_px=args.tile_h_px,
        min_coverage=args.min_coverage,
    )
    octomap._sync_from_grid_handler()
    octomap.build_octomap(octomap.grid_handler.blocked_obstacles)

    occ2d = octomap.build_occ2d(use_columns=bool(args.use_columns))
    z_band2d = octomap.build_z_band2d()

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        occ2d=occ2d,
        z_band2d=z_band2d,
        grid_w=np.int32(octomap.grid_w),
        grid_h=np.int32(octomap.grid_h),
        grid_scale=np.int32(octomap.grid_scale),
        revision=np.int32(octomap.revision),
    )
    print(f"saved: {out}")

    if args.preview:
        z_lo = z_band2d[..., 0]
        z_hi = z_band2d[..., 1]
        thickness = np.maximum(0.0, z_hi - z_lo)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(occ2d, cmap="gray")
        axes[0].set_title("occ2d")
        axes[1].imshow(z_hi, cmap="magma")
        axes[1].set_title("z_hi (top_z)")
        axes[2].imshow(thickness, cmap="viridis")
        axes[2].set_title("z_band_thickness")
        for ax in axes:
            ax.set_axis_off()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
