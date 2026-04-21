# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

- Local business logic is now centralized under `app/`.
- Redundant legacy wrappers under `models/`, `tools/`, and most top-level `utils/` have been removed.
- `yolo/` remains a vendored YOLOv5 segmentation codebase.
- The main workflow is: segmentation inference → `runs/<exp>/masks` → grid map extraction → D* Lite path planning → interactive visualization.

## Architecture

- `app/inference/segmentation.py` is the standard inference CLI. It builds a subprocess command for `yolo/segment/predict.py`, applies repo-relative defaults from `app/config.py` and `app/paths.py`, and can reload generated masks.
- `yolo/segment/predict.py` is still the key integration point inside vendored code. It writes one mask per detection under `runs/.../masks` using the required filename pattern `<cls_id>_<image_stem>_<index>.png`.
- `app/mapping/grid_map.py` converts mask rasters into occupancy cells. Obstacle classes are `{1, 3, 4, 5, 6, 9}` and target class is `0`; the target goal is the centroid of the target mask.
- `app/planning/dstar_lite.py` contains the custom D* Lite planner with obstacle inflation and a penalty map so routes prefer safer clearance.
- `app/planning/path_planner.py` is the standard interactive planner entrypoint. It loads masks, builds the grid, chooses the goal, and replans on mouse interaction.
- `app/tooling/json_group_fix.py` and `app/tooling/rename_images.py` hold the standalone utility CLIs.
- `utils/save.py` is a small remaining local utility for run/output directory naming.
- `data/my.yaml` currently contains class metadata only (`nc` + `names`). It is suitable for inference metadata but not for upstream YOLO train/val commands unless replaced with a full dataset YAML that includes `train:` and `val:`.

## Common commands

Run from the repository root.

- Install dependencies:
  - `python -m pip install -r requirements.txt`
- Run inference:
  - `python -m app.inference.segmentation --source test_input --load-masks`
- Launch the interactive planner:
  - `python -m app.planning.path_planner --mask-dir runs/exp/masks`
- Fix `group_id` values inside annotation JSON files:
  - `python -m app.tooling.json_group_fix path/to/json_dir`
- Bulk rename images:
  - `python -m app.tooling.rename_images path/to/images --prefix group6 --start-num 1 --padding 6`
- Run tests:
  - `python -m pytest`
- Run a single test file:
  - `python -m pytest tests/test_grid_map.py -q`
- Run formatting / linting for local code only:
  - `python -m black app tests utils`
  - `python -m ruff check app tests utils`

## Important implementation assumptions

- Prefer editing files under `app/`. They are now the only source of truth for local business logic.
- Do not change the mask filename convention `<cls_id>_<image_stem>_<index>.png` unless you also update downstream parsing in `app/mapping/grid_map.py` and related planning code.
- Path defaults are centralized in `app/paths.py` and `app/config.py`. The intended precedence is CLI args > environment variables > repo-relative defaults.
- `yolo/` should be treated as vendored upstream code. Avoid broad formatting or refactors there unless the task specifically targets the vendored implementation.
