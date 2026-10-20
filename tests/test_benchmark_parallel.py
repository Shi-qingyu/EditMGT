"""CPU-only regression cases for session sharding and global output merging."""
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from eval.benchmark_magicbrush import parse_args
from eval.benchmark_inter_edit import parse_args as parse_inter_edit_args
from src import benchmark_parallel as parallel
from src import benchmark_utils
from src.benchmark_data import BenchmarkImages, magicbrush_tasks, path_spec
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


@pytest.mark.parametrize("num_workers", [1, 2, 8, 20])
def test_inter_edit_shards_cover_each_sample_once_with_balanced_loads(num_workers):
    selection = [{"key": f"sample_{index:05d}"} for index in range(11)]
    shards = parallel.shard_inter_edit_samples(selection, num_workers)
    assert len(shards) == min(num_workers, len(selection))
    assigned = [task["key"] for shard in shards for task in shard]
    assert len(assigned) == len(set(assigned)) == len(selection)
    assert set(assigned) == {task["key"] for task in selection}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    for shard in shards:
        assert shard == sorted(shard, key=lambda task: task["key"])


def test_inter_edit_sharding_rejects_empty_selection_and_invalid_worker_count():
    with pytest.raises(ValueError, match="empty"):
        parallel.shard_inter_edit_samples([], 8)
    with pytest.raises(ValueError, match="positive"):
        parallel.shard_inter_edit_samples([{"key": "sample_00000"}], 0)


def test_inter_edit_parser_rejects_invalid_gpu_count(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_inter_edit.py", "--vanilla", "--num-gpus", "0"])
    with pytest.raises(SystemExit) as error:
        parse_inter_edit_args()
    assert error.value.code == 2


@pytest.mark.parametrize("native", [False, True])
def test_inter_edit_parallel_scoring_resume_and_failure(tmp_path, monkeypatch, native):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "transformer").mkdir(parents=True)
    (checkpoint / "transformer/config.json").write_text("{}")
    # Native argument defaults need only this constant, not model imports.
    monkeypatch.setitem(sys.modules, "src.v2_model", SimpleNamespace(negative_prompt="original negative prompt"))
    monkeypatch.setattr(sys, "argv", [
        "benchmark_inter_edit.py", "--metrics", "l1", "--num-gpus", "8",
        "--metric-device", "cpu", "--tar-cache-dir", str(tmp_path / "tar-cache"),
        "--output-dir", str(tmp_path / "outputs"),
        *([] if native else ["--checkpoint", str(checkpoint)]),
    ])
    args = parse_inter_edit_args(native_editmgt=native)
    selection = []
    for index in range(11):
        specs = {}
        for name, value in (("source", 0), ("target", 20 if index == 10 else 10), ("mask", 255)):
            path = tmp_path / f"{name}-{index}.png"
            Image.new("RGB", (4, 4), (value,) * 3).save(path)
            specs[name] = path_spec(path)
        selection.append({
            "key": f"sample_{index:05d}", "seed_key": f"inter_edit:{index}",
            "instruction": "add ten", "language": "en", "edit_type": "test",
            "prediction": f"images/{index:05d}.png", **specs,
        })
    dataset = {"split": "test", "language": "en"}
    phases, generated, caches = [], [], []

    class FakeRunner:
        metadata = {"device_name": "fake GPU"}

        def __init__(self, worker_args):
            assert worker_args.native_editmgt == native

        def generate(self, source, mask, instruction, seed):
            assert (mask is None) == native
            generated.append(seed)
            return Image.new("RGB", source.size, (10,) * 3), {
                "seed": seed, "inference_seconds": 1.0, "execution": self.metadata,
            }

        def close(self):
            pass

    def tracked_images(tar_cache_dir):
        caches.append(tar_cache_dir)
        return BenchmarkImages(tar_cache_dir)

    def fake_spawn(worker_args, run, shards, names, worker_dirs, devices, metric_devices):
        phases.append(worker_args.stage)
        with pytest.raises(BlockingIOError):
            BenchmarkRun(worker_args, "inter_edit", selection, dataset)
        for rank in range(len(shards)):
            parallel._benchmark_worker(rank, worker_args, run.root, run.info, shards,
                                       names, worker_dirs, devices, metric_devices)

    monkeypatch.setattr(benchmark_utils, "InferenceRunner", FakeRunner)
    monkeypatch.setattr(parallel, "BenchmarkImages", tracked_images)
    monkeypatch.setattr(parallel, "_spawn_workers", fake_spawn)
    monkeypatch.setattr(parallel.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(parallel.torch.cuda, "device_count", lambda: 8)
    run = BenchmarkRun(args, "inter_edit", selection, dataset)
    try:
        assert parallel.run_inter_edit_data_parallel(args, run, ("l1",))
    finally:
        run.close()
    assert phases == ["generate", "metrics"]
    assert len(generated) == len(selection)
    assert caches == [args.tar_cache_dir] * 16
    summary = json.loads((args.output_dir / "metrics/summary.json").read_text())
    assert summary["complete"] and summary["scored_predictions"] == 11
    assert summary["overall"]["l1"]["mean"] == pytest.approx(10 / (255 * 11))
    assert summary["by_language"]["en"] == summary["overall"]
    assert summary["by_edit_type"]["test"] == summary["overall"]
    assert summary["data_parallel"]["world_size"] == 8
    assert summary["data_parallel"]["partition"] == "independent_inter_edit_samples"
    score_file = args.output_dir / "metrics/per_sample.jsonl"
    scores = [json.loads(line) for line in score_file.read_text().splitlines()]
    assert [row["key"] for row in scores] == [task["key"] for task in selection]

    args.resume, args.num_gpus = True, 2
    generated.clear()
    run = BenchmarkRun(args, "inter_edit", selection, dataset)
    try:
        assert parallel.run_inter_edit_data_parallel(args, run, ("l1",))
    finally:
        run.close()
    assert not generated
    assert [json.loads(line) for line in score_file.read_text().splitlines()] == scores

    args.stage = "metrics"
    run = BenchmarkRun(args, "inter_edit")
    phases.clear()
    try:
        assert parallel.run_inter_edit_data_parallel(args, run, ("l1",))
    finally:
        run.close()
    assert phases == ["metrics"] and not generated

    def fail_spawn(*unused):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(parallel, "_spawn_workers", fail_spawn)
    run = BenchmarkRun(args, "inter_edit")
    try:
        with pytest.raises(RuntimeError, match="worker failed"):
            parallel.run_inter_edit_data_parallel(args, run, ("l1",))
    finally:
        run.close()
    failed = json.loads((args.output_dir / "metrics/summary.json").read_text())
    assert not failed["complete"] and failed["scored_predictions"] == 0
    assert failed["unprocessed_predictions"] == len(selection)


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_inter_edit_missing_and_empty_masks_are_skipped_and_resume_reuses_valid_images(tmp_path, monkeypatch, num_gpus):
    monkeypatch.setattr(sys, "argv", [
        "benchmark_inter_edit.py", "--vanilla", "--metrics", "l1",
        "--num-gpus", str(num_gpus), "--metric-device", "cpu", "--resolution", "32",
        "--output-dir", str(tmp_path / "outputs"),
    ])
    args = parse_inter_edit_args()
    source, target = tmp_path / "source.png", tmp_path / "target.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(source)
    Image.new("RGB", (4, 4), (10, 10, 10)).save(target)
    nonempty, empty = tmp_path / "mask.png", tmp_path / "empty.png"
    Image.new("L", (4, 4), 255).save(nonempty)
    Image.new("L", (4, 4), 0).save(empty)
    masks = [path_spec(nonempty), path_spec(empty), None, path_spec(tmp_path / "absent.png"),
             {"kind": "tar", "root": str(tmp_path), "archive": "masks.tar", "member": None}]
    selection = [{
        "key": f"sample_{index:05d}", "test_index": index, "seed_key": f"inter_edit:{index}",
        "instruction": "make it gray", "language": "en", "edit_type": "test",
        "prediction": f"images/{index:05d}.png", "source": path_spec(source),
        "target": path_spec(target), "mask": mask,
    } for index, mask in enumerate(masks)]
    generated = []

    class FakeRunner:
        metadata = {"device_name": "fake GPU"}

        def __init__(self, worker_args):
            pass

        def generate(self, source, mask, instruction, seed):
            assert mask is not None and mask.getbbox() is not None
            generated.append(seed)
            return Image.new("RGB", source.size, (10, 10, 10)), {
                "seed": seed, "inference_seconds": 1.0, "execution": self.metadata,
            }

        def close(self):
            pass

    def fake_spawn(worker_args, run, shards, names, worker_dirs, devices, metric_devices):
        for rank in range(len(shards)):
            parallel._benchmark_worker(rank, worker_args, run.root, run.info, shards,
                                       names, worker_dirs, devices, metric_devices)

    monkeypatch.setattr(benchmark_utils, "InferenceRunner", FakeRunner)
    monkeypatch.setattr(parallel, "_spawn_workers", fake_spawn)
    monkeypatch.setattr(parallel.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(parallel.torch.cuda, "device_count", lambda: 2)
    dataset = {"split": "test", "language": "en"}
    for attempt in range(2):
        args.resume = attempt > 0
        run = BenchmarkRun(args, "inter_edit", selection, dataset)
        if attempt == 0:
            # An error record from the previous implementation must become a
            # skip, without invalidating the saved generation settings/selection.
            benchmark_utils.atomic_json(run.record_path(selection[1]["key"]),
                                        {"status": "error", "error": "Empty benchmark mask"})
        try:
            if num_gpus > 1:
                assert parallel.run_inter_edit_data_parallel(args, run, ("l1",))
            else:
                images = BenchmarkImages()
                try:
                    benchmark_utils.generate_benchmark(args, run, images)
                    assert benchmark_utils.score_benchmark(args, run, images, ("l1",))
                finally:
                    images.close()
            assert [run.record(task["key"])["status"] for task in selection] == ["ok"] + ["skipped"] * 4
        finally:
            run.close()
        summary = json.loads((args.output_dir / "metrics/summary.json").read_text())
        assert summary["selected_predictions"] == 5
        assert summary["expected_predictions"] == summary["scored_predictions"] == 1
        assert summary["skipped_predictions"] == 4
        assert summary["skipped_by_reason"] == {"empty_mask": 1, "missing_mask": 3}
        assert summary["complete"] and not summary["full_selection_scored"]
        assert summary["failed_predictions"] == summary["unprocessed_predictions"] == 0
        assert summary["timing"]["count"] == 1
        assert len(generated) == 1  # The resume iteration never calls inference again.
        assert len(list((args.output_dir / "images").glob("*.png"))) == 1
        assert json.loads((args.output_dir / "selection.json").read_text()) == selection


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
