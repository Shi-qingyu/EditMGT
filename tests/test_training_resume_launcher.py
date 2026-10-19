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
        "'halo': os.environ['SPARSE_TARGET_HALO'], 'time': os.environ['TIME_CONDITIONING']}))"
    )
    (project / "scripts/train_en_cache_magicbrush.sh").write_text(
        "#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable) + " -c "
        + shlex.quote(capture) + ' "$@"\n'
    )
    legacy = project / "runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation/checkpoint-2600"
    (legacy / "transformer").mkdir(parents=True)
    for filename in ("transformer/config.json", "optimizer.bin", "scheduler.bin"):
        (legacy / filename).write_text("test fixture")
    target = project / "runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix"
    env = dict(os.environ)
    env.pop("TRAIN_OUTPUT_DIR", None)
    env.pop("RESUME_FROM_CHECKPOINT", None)
    return launcher, legacy, target, env


def _capture(launcher, env, *args):
    result = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_first_launch_resumes_2600_and_overrides_inherited_fresh_start(resume_launcher):
    launcher, legacy, target, env = resume_launcher
    captured = _capture(launcher, env, "--fresh_start", "--max_train_steps", "4000")
    args = captured["args"]
    assert args[args.index("--resume_from_checkpoint") + 1] == str(legacy)
    assert args[args.index("--output_dir") + 1] == captured["output"] == str(target)
    assert args.index("--no-fresh_start") > args.index("--fresh_start")
    assert "--require_resume" in args and "--no-supervise_target_halo" in args
    assert args[args.index("--max_train_steps") + 1] == "4000"
    assert captured["halo"] == "0" and captured["time"] == "mask_ratio"
    assert not target.exists()


def test_relaunch_uses_latest_corrected_checkpoint_without_rewinding(resume_launcher):
    launcher, legacy, target, env = resume_launcher
    (target / "checkpoint-2800").mkdir(parents=True)
    marker = target / "run_initialization.json"
    marker.write_text("original resume provenance")
    (legacy / "optimizer.bin").unlink()
    captured = _capture(launcher, env)
    args = captured["args"]
    assert args[args.index("--resume_from_checkpoint") + 1] == "latest"
    assert marker.read_text() == "original resume provenance"


def test_explicit_resume_checkpoint_takes_precedence_over_latest(resume_launcher):
    launcher, legacy, target, env = resume_launcher
    (target / "checkpoint-2800").mkdir(parents=True)
    env["RESUME_FROM_CHECKPOINT"] = str(legacy)
    captured = _capture(launcher, env)
    args = captured["args"]
    assert args[args.index("--resume_from_checkpoint") + 1] == str(legacy)


def test_incomplete_source_checkpoint_is_rejected_before_launch(resume_launcher):
    launcher, legacy, target, env = resume_launcher
    (legacy / "optimizer.bin").unlink()
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "missing optimizer.bin" in result.stderr
    assert not result.stdout and not target.exists()
