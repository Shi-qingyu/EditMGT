"""Check resume-launcher arguments without starting Python training or CUDA."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.fixture
def resume_launcher(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    name = "train_en_cache_magicbrush_no_dilation_resume.sh"
    source = Path(__file__).resolve().parents[1] / "scripts" / name
    launcher = project / "scripts" / name
    launcher.write_text(source.read_text())
    # Capture the shared launcher's input; no training code is invoked.
    capture = (
        "import json, os, sys; "
        "print(json.dumps({'args': sys.argv[1:], 'output': os.environ['TRAIN_OUTPUT_DIR'], "
        "'halo': os.environ['SPARSE_TARGET_HALO'], 'time': os.environ['TIME_CONDITIONING'], "
        "'bbox_prob': os.environ['MASK_BBOX_PROB'], 'cache_threshold': os.environ['CONDITION_CACHE_THRESHOLD'], "
        "'skip_install': os.environ['SKIP_INSTALL']}))"
    )
    (project / "scripts/train_en_cache_magicbrush.sh").write_text(
        "#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable) + " -c "
        + shlex.quote(capture) + ' "$@"\n'
    )
    target = project / "runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix"
    source_checkpoint = target / "checkpoint-6800"
    (source_checkpoint / "transformer").mkdir(parents=True)
    for filename in ("transformer/config.json", "optimizer.bin", "scheduler.bin"):
        (source_checkpoint / filename).write_text("test fixture")
    env = dict(os.environ)
    env.pop("TRAIN_OUTPUT_DIR", None)
    env.pop("RESUME_FROM_CHECKPOINT", None)
    env.pop("SKIP_INSTALL", None)
    env.pop("INTER_EDIT_TRANSLATIONS_DIR", None)
    env.pop("MASKEDIT_DATA_DIR", None)
    env.pop("MASKEDIT_PROBABILITY", None)
    env.pop("MASK_BBOX_PROB", None)
    env.pop("CONDITION_CACHE_THRESHOLD", None)
    return launcher, source_checkpoint, target, env


def _capture(launcher, env, *args):
    result = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_first_launch_resumes_6800_and_overrides_inherited_fresh_start(resume_launcher):
    launcher, source_checkpoint, target, env = resume_launcher
    captured = _capture(launcher, env, "--fresh_start", "--max_train_steps", "10000")
    args = captured["args"]
    assert args[args.index("--resume_from_checkpoint") + 1] == str(source_checkpoint)
    assert args[args.index("--output_dir") + 1] == captured["output"] == str(target)
    assert args.index("--no-fresh_start") > args.index("--fresh_start")
    assert "--require_resume" in args and "--no-supervise_target_halo" in args
    assert args[args.index("--max_train_steps") + 1] == "10000"
    assert captured["halo"] == "0" and captured["time"] == "mask_ratio"
    project = launcher.parents[1]
    assert args[args.index("--inter_edit_translations_dir") + 1] == str(project / "data/inter_edit_zh_to_en_qwen35_9b")
    assert args[args.index("--maskedit_data_dir") + 1] == str(project / "data/MaskEdit-10K")
    assert args[args.index("--maskedit_probability") + 1] == "0.1"
    assert captured["skip_install"] == "1"
    assert captured["bbox_prob"] == "0.1"
    assert captured["cache_threshold"] == "0.8"
    assert "--allow_condition_cache_threshold_change_on_resume" in args


def test_explicit_bbox_probability_overrides_resume_default(resume_launcher):
    launcher, _, _, env = resume_launcher
    env["MASK_BBOX_PROB"] = "0.25"
    assert _capture(launcher, env)["bbox_prob"] == "0.25"


def test_explicit_cache_threshold_overrides_resume_default(resume_launcher):
    launcher, _, _, env = resume_launcher
    env["CONDITION_CACHE_THRESHOLD"] = "0.6"
    assert _capture(launcher, env)["cache_threshold"] == "0.6"


def test_relaunch_uses_latest_corrected_checkpoint_without_rewinding(resume_launcher):
    launcher, source_checkpoint, target, env = resume_launcher
    (target / "checkpoint-7000").mkdir(parents=True)
    marker = target / "run_initialization.json"
    marker.write_text("original resume provenance")
    (source_checkpoint / "optimizer.bin").unlink()
    captured = _capture(launcher, env)
    args = captured["args"]
    assert args[args.index("--resume_from_checkpoint") + 1] == "latest"
    assert marker.read_text() == "original resume provenance"


def test_explicit_resume_checkpoint_takes_precedence_over_latest(resume_launcher):
    launcher, source_checkpoint, target, env = resume_launcher
    (target / "checkpoint-7000").mkdir(parents=True)
    env["RESUME_FROM_CHECKPOINT"] = str(source_checkpoint)
    captured = _capture(launcher, env)
    args = captured["args"]
    assert args[args.index("--resume_from_checkpoint") + 1] == str(source_checkpoint)


def test_incomplete_source_checkpoint_is_rejected_before_launch(resume_launcher):
    launcher, source_checkpoint, target, env = resume_launcher
    (source_checkpoint / "optimizer.bin").unlink()
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "missing optimizer.bin" in result.stderr
    assert not result.stdout
