"""CPU-only regression cases for session sharding and global output merging."""
import json
import sys

import pytest
from PIL import Image

from eval.benchmark_magicbrush import parse_args
from src import benchmark_parallel as parallel
from src import benchmark_utils
from src.benchmark_data import magicbrush_tasks, path_spec
from src.benchmark_utils import BenchmarkRun


def _sessions(lengths):
    return {
        f"session-{index}": [
            {"session_id": f"session-{index}", "turn": turn, "instruction": "add ten"}
            for turn in range(1, length + 1)
        ]
        for index, length in enumerate(lengths)
    }


@pytest.mark.parametrize("mode", ["single", "multi", "both"])
def test_shards_have_no_padding_and_preserve_whole_sessions_and_dependencies(mode):
    selection, _ = magicbrush_tasks(_sessions([1, 3, 2, 3, 1, 2, 1, 3, 2, 1, 3]), mode)
    shards = parallel.shard_magicbrush_sessions(selection, 8)
    assert len(shards) == 8
    assigned = [task["key"] for shard in shards for task in shard]
    assert len(assigned) == len(set(assigned)) == len(selection)
    assert set(assigned) == {task["key"] for task in selection}
    owners = {}
    for rank, shard in enumerate(shards):
        seen = set()
        for task in shard:
            assert owners.setdefault(task["session_id"], rank) == rank
            assert task.get("input_prediction_key") is None or task["input_prediction_key"] in seen
            seen.add(task["key"])
        assert shard == [task for task in selection if task["key"] in seen]
    small, _ = magicbrush_tasks(_sessions([3, 1]), mode)
    assert len(parallel.shard_magicbrush_sessions(small, 8)) == 2


def test_sharding_rejects_missing_multi_turn_predecessor():
    selection, _ = magicbrush_tasks(_sessions([2]), "multi")
    with pytest.raises(ValueError, match="Multi-turn dependency"):
        parallel.shard_magicbrush_sessions(selection[1:], 8)


def test_parallel_generation_scoring_resume_and_failed_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "benchmark_magicbrush.py", "--vanilla", "--mask-mode", "none",
        "--metrics", "l1", "--num-gpus", "2", "--output-dir", str(tmp_path / "outputs"),
    ])
    args = parse_args()
    sessions = _sessions([3, 1, 1])
    for turns in sessions.values():
        for task in turns:
            source = tmp_path / f"{task['session_id']}-source-{task['turn']}.png"
            target = tmp_path / f"{task['session_id']}-target-{task['turn']}.png"
            Image.new("RGB", (4, 4), (20 * (task["turn"] - 1),) * 3).save(source)
            Image.new("RGB", (4, 4), (10 * task["turn"],) * 3).save(target)
            task.update(source=path_spec(source), target=path_spec(target), mask=None)
    selection, _ = magicbrush_tasks(sessions)
    dataset = {"split": "dev"}
    phases, generated = [], []

    class FakeRunner:
        metadata = {"device_name": "fake GPU"}

        def __init__(self, worker_args):
            self.device = worker_args.device

        def generate(self, source, mask, instruction, seed):
            assert mask is None
            generated.append((self.device, seed))
            value = source.getpixel((0, 0))[0] + 10
            return Image.new("RGB", source.size, (value,) * 3), {
                "seed": seed, "inference_seconds": 1.0, "execution": self.metadata,
            }

        def close(self):
            pass

    def fake_spawn(worker_args, run, shards, names, worker_dirs, devices, metric_devices):
        phases.append(worker_args.stage)
        # The parent must retain the directory lock through both phases.
        with pytest.raises(BlockingIOError):
            BenchmarkRun(worker_args, "magicbrush", selection, dataset)
        for rank in range(len(shards)):
            parallel._benchmark_worker(rank, worker_args, run.root, run.info, shards,
                                       names, worker_dirs, devices, metric_devices)

    monkeypatch.setattr(benchmark_utils, "InferenceRunner", FakeRunner)
    monkeypatch.setattr(parallel, "_spawn_workers", fake_spawn)
    monkeypatch.setattr(parallel.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(parallel, "_worker_devices", lambda worker_args, count: (
        [f"cuda:{rank}" for rank in range(count)], ["cpu"] * count,
    ))
    run = BenchmarkRun(args, "magicbrush", selection, dataset)
    try:
        assert parallel.run_magicbrush_data_parallel(args, run, ("l1",))
    finally:
        run.close()
    assert phases == ["generate", "metrics"]
    assert len(generated) == len(selection) == 7
    summary = json.loads((args.output_dir / "metrics/summary.json").read_text())
    assert summary["scored_predictions"] == 7 and summary["complete"]
    single = summary["by_protocol"]["single_turn_all"]
    assert single["scored"] == 5
    # The two worker shards have 5 and 2 predictions: use per-image weighting.
    assert single["metrics"]["l1"]["mean"] == pytest.approx(30 / (255 * 5))
    multi = summary["by_protocol"]["multi_turn_final"]
    assert multi["scored"] == 3 and multi["metrics"]["l1"]["mean"] == 0
    scores = [json.loads(line) for line in (args.output_dir / "metrics/per_sample.jsonl").read_text().splitlines()]
    assert [row["key"] for row in scores] == [task["key"] for task in selection]

    # A differently sized resume skips every saved edit, keeping original seeds.
    args.resume, args.num_gpus = True, 8
    generated.clear()
    run = BenchmarkRun(args, "magicbrush", selection, dataset)
    try:
        assert parallel.run_magicbrush_data_parallel(args, run, ("l1",))
    finally:
        run.close()
    assert not generated
    resumed = [json.loads(line) for line in (args.output_dir / "metrics/per_sample.jsonl").read_text().splitlines()]
    assert resumed == scores

    # A failed new scoring launch must publish an incomplete result, rather
    # than merging old worker output or leaving the prior complete summary.
    def fail_spawn(*unused):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(parallel, "_spawn_workers", fail_spawn)
    args.stage = "metrics"
    run = BenchmarkRun(args, "magicbrush")
    try:
        with pytest.raises(RuntimeError, match="worker failed"):
            parallel.run_magicbrush_data_parallel(args, run, ("l1",))
    finally:
        run.close()
    failed = json.loads((args.output_dir / "metrics/summary.json").read_text())
    assert not failed["complete"] and failed["scored_predictions"] == 0
    assert failed["unprocessed_predictions"] == len(selection)
