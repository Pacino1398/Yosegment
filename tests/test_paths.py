import sys
from pathlib import Path

from app.inference.segmentation import build_predict_command
from app.paths import get_default_mask_dir, resolve_path
from app.planning.pathplan_batch import (
    create_pathplan_run_dir,
    get_frame_stem,
    get_latest_segmentation_run_dir,
    get_next_segmentation_run_dir,
)


def test_build_predict_command_uses_explicit_arguments(tmp_path):
    weights = tmp_path / "model.pt"
    data_yaml = tmp_path / "data.yaml"
    source = tmp_path / "images"
    project = tmp_path / "runs"

    command = build_predict_command(
        weights=weights,
        data_yaml=data_yaml,
        source=source,
        project=project,
        name="demo",
        device="cpu",
        conf_thres=0.4,
    )

    assert command[0] == sys.executable
    assert command[1].endswith("yolo/segment/predict.py")
    assert command[command.index("--weights") + 1] == str(weights.resolve())
    assert command[command.index("--data") + 1] == str(data_yaml.resolve())
    assert command[command.index("--source") + 1] == str(source.resolve())
    assert command[command.index("--project") + 1] == str(project.resolve())
    assert command[command.index("--name") + 1] == "demo"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--conf-thres") + 1] == "0.4"


def test_resolve_path_and_default_mask_dir():
    default = Path("D:/example/default")

    assert resolve_path(None, default) == default
    assert resolve_path(".", default).is_absolute()
    assert get_default_mask_dir("exp42").name == "masks"
    assert get_default_mask_dir("exp42").parent.name == "exp42"


def test_create_pathplan_run_dir_uses_exp_then_exp1_then_exp2(tmp_path):
    project_dir = tmp_path / "pathplan"

    first = create_pathplan_run_dir(project_dir)
    second = create_pathplan_run_dir(project_dir)
    third = create_pathplan_run_dir(project_dir)

    assert first.name == "exp"
    assert second.name == "exp1"
    assert third.name == "exp2"


def test_get_next_segmentation_run_dir_uses_exp_then_exp1_then_exp2(tmp_path):
    project_dir = tmp_path / "segment"

    first = get_next_segmentation_run_dir(project_dir)
    first.mkdir(parents=True)
    second = get_next_segmentation_run_dir(project_dir)
    second.mkdir(parents=True)
    third = get_next_segmentation_run_dir(project_dir)

    assert first.name == "exp"
    assert second.name == "exp1"
    assert third.name == "exp2"


def test_get_latest_segmentation_run_dir_returns_highest_index(tmp_path):
    project_dir = tmp_path / "segment"
    (project_dir / "exp").mkdir(parents=True)
    (project_dir / "exp1").mkdir(parents=True)
    (project_dir / "exp3").mkdir(parents=True)

    latest = get_latest_segmentation_run_dir(project_dir)

    assert latest.name == "exp3"


def test_get_frame_stem_matches_video_mask_group_name():
    assert get_frame_stem(Path("demo.mp4"), 123) == "demo_frame000123"
