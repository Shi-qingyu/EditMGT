"""CPU tests for durable resume and real spawned-worker coordination, using fake vLLM."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from scripts import translate_inter_edit as translation


FAKE_VLLM = r'''
import json
import os
import time
from types import SimpleNamespace

def event(value):
    fd = os.open(os.environ["FAKE_VLLM_EVENTS"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps(value, ensure_ascii=False) + "\n").encode())
    finally:
        os.close(fd)

class SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return messages[-1]["content"]

class LLM:
    def __init__(self, **kwargs):
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        assert len(devices) == kwargs["tensor_parallel_size"]
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        event({"type": "init", "devices": devices, "kwargs": kwargs})

    def get_tokenizer(self):
        return Tokenizer()

    def generate(self, prompts, params, use_tqdm):
        event({"type": "generate", "prompts": prompts})
        time.sleep(float(os.environ.get("FAKE_VLLM_DELAY", "0.02")))
        return [SimpleNamespace(outputs=[SimpleNamespace(
            text="" if prompt == os.environ.get("FAKE_FAIL_PROMPT") else "Add object " + prompt.split()[-1] + ".",
            finish_reason="stop",
        )]) for prompt in prompts]
'''


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "Inter-Edit"
        self.train = self.data / "train"
        self.metadata = self.train / "metadata"
        self.metadata.mkdir(parents=True)
        self.output = self.root / "translated"
        (self.train / "manifest.json").write_text(json.dumps({"num_sample_shards": 4}))
        for shard in range(4):
            rows = [
                {"sample_id": shard * 3, "instruction": f"添加物体 {shard * 3}"},
                {"sample_id": shard * 3 + 1, "instruction": "Add an object."},
                {"sample_id": shard * 3 + 2, "instruction": f"把红色 object 改为蓝色 {shard * 3 + 2}"},
            ]
            with gzip.open(self.metadata / f"train-{shard:05d}.jsonl.gz", "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.paths = sorted(self.metadata.glob("*.gz"))

    def args(self, *extra):
        return translation.parse_args(["--data-dir", str(self.data), "--output-dir", str(self.output),
                                       "--batch-size", "1", *extra])

    def prepare_output(self):
        for directory in ("shards", "errors"):
            (self.output / directory).mkdir(parents=True, exist_ok=True)

    def create_test_metadata(self, jsonl_only=False):
        directory = self.data / "test"
        directory.mkdir(exist_ok=True)
        rows = [{"index": index, "language": "中文" if index % 2 == 0 else "English",
                 "instruction": f"添加物体 {index}" if index % 2 == 0 else 'Write "你好" on the sign.',
                 "edit_type": "Add", "source_path": f"source_images/{index:05d}.jpg",
                 "gt_path": f"target_images/{index:05d}.png", "mask_path": f"masks/{index:05d}.png"}
                for index in range(16)]
        path = directory / ("metadata.jsonl" if jsonl_only else "metadata.json")
        if jsonl_only:
            # Physical JSONL line numbers must survive blank lines and chunking.
            path.write_text("\n\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        else:
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return path, rows

    def test_test_split_uses_labels_and_stable_chunks_instead_of_train_shards(self):
        source, original = self.create_test_metadata()
        args = self.args("--split", "test", "--test-shard-size", "1", "--dry-run")
        root, paths = translation.discover_inputs(self.data, "test")
        self.assertEqual((root, paths), (source.parent, [source]))
        tasks, count, done = translation.make_plan(args, paths)
        self.assertEqual((len(tasks), count, done), (8, 8, 0))
        self.assertEqual(len({task["checkpoint_name"] for task in tasks}), 8)
        rows = [row for task in tasks for row in translation.task_rows(args, task)]
        self.assertEqual([row["sample_id"] for row in rows], list(range(0, 16, 2)))
        self.assertTrue(all(row["metadata_file"] == "metadata.json" for row in rows))
        self.assertFalse(self.output.exists())
        self.assertFalse(translation.needs_translation(original[1], "test"))
        self.assertTrue(translation.needs_translation({"language": "CN", "instruction": "iPhone"}, "test"))
        config = translation.run_config(args, root, paths)
        self.assertEqual(config["source_test_dir"], str(root))
        self.assertNotIn("source_train_dir", config)
        train_config = translation.run_config(self.args(), self.train, self.paths)
        self.assertEqual(train_config["format_version"], 1)
        self.assertNotIn("split", train_config)
        self.assertNotIn("test_shard_size", train_config)

    def test_jsonl_test_chunks_preserve_physical_line_indices(self):
        source, _ = self.create_test_metadata(jsonl_only=True)
        args = self.args("--split", "test", "--test-shard-size", "3", "--dry-run")
        _, paths = translation.discover_inputs(self.data / "test", "test")
        self.assertEqual(paths, [source])
        tasks, count, _ = translation.make_plan(args, paths)
        selected = [row for task in tasks for row in translation.task_rows(args, task)]
        self.assertEqual(count, 8)
        self.assertEqual([row["row_index"] for row in selected], list(range(0, 32, 4)))
        self.assertEqual([task["count"] for task in tasks], [3, 3, 2])

    def test_limited_test_translation_does_not_export_a_partly_english_dataset(self):
        source, _ = self.create_test_metadata()
        self.prepare_output()
        args = self.args("--split", "test", "--test-shard-size", "2", "--limit", "3")
        tasks, total, _ = translation.make_plan(args, [source])

        class Translator:
            def translate(self, rows, attempt):
                return [(f"Add object {row['sample_id']}.", "stop") for row in rows]

        for task in tasks:
            translation.process_shard(args, task, Translator())
        summary = translation.merge_results(args, tasks, total, test_source=source)
        self.assertEqual((summary["translated"], summary["complete"]), (3, True))
        self.assertIsNone(summary["metadata_file"])
        self.assertFalse((self.output / "metadata_en.json").exists())

    def test_eight_test_replicas_resume_and_export_all_rows_without_changing_english(self):
        source, original = self.create_test_metadata()
        source_bytes = source.read_bytes()
        fake_dir = self.root / "fake_modules"
        fake_dir.mkdir()
        (fake_dir / "vllm.py").write_text(FAKE_VLLM)
        events_path = self.root / "test_events.jsonl"
        env = {**os.environ, "PYTHONPATH": str(fake_dir), "FAKE_VLLM_EVENTS": str(events_path),
               "FAKE_FAIL_PROMPT": "添加物体 0"}
        command = [sys.executable, str(Path(translation.__file__)), "--data-dir", str(self.data),
                   "--split", "test", "--test-shard-size", "1", "--batch-size", "1", "--retries", "0",
                   "--gpu-ids", "0,1,2,3,4,5,6,7",
                   "--output-dir", str(self.output)]
        result = subprocess.run(command + ["--num-gpus", "8"], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((self.output / "metadata_en.json").exists())
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["pending"], 1)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        self.assertEqual(len([item for item in events if item["type"] == "init"]), 8)
        self.assertEqual(len(list((self.output / "errors").glob("*.errors.json"))), 8)
        count = len(events)
        del env["FAKE_FAIL_PROMPT"]
        result = subprocess.run(command + ["--num-gpus", "1"], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        self.assertEqual([item["prompts"] for item in events[count:] if item["type"] == "generate"], [["添加物体 0"]])
        exported = json.loads((self.output / "metadata_en.json").read_text())
        self.assertEqual(len(exported), len(original))
        self.assertEqual([row["index"] for row in exported], list(range(16)))
        self.assertTrue(all(row["language"] == "English" for row in exported))
        for index, row in enumerate(exported):
            self.assertEqual(row["source_path"], original[index]["source_path"])
            if index % 2:
                self.assertEqual(row, original[index])
            else:
                self.assertEqual(row["instruction_original"], original[index]["instruction"])
                self.assertEqual(row["language_original"], "中文")
        self.assertEqual([json.loads(line) for line in (self.output / "metadata_en.jsonl").read_text().splitlines()], exported)
        self.assertEqual(source.read_bytes(), source_bytes)
        args = self.args("--split", "test", "--test-shard-size", "2")
        with self.assertRaisesRegex(ValueError, "test_shard_size"):
            translation.check_config(self.output, translation.run_config(args, source.parent, [source]))

    def test_selection_limits_and_read_only_dry_run_on_real_format(self):
        self.assertEqual(translation.discover_inputs(self.data)[0], self.train)
        self.assertEqual(translation.discover_inputs(self.train)[1], self.paths)
        rows = translation.selected_rows(self.paths[0])
        self.assertEqual([row["sample_id"] for row in rows], [0, 2])
        args = self.args("--dry-run", "--limit", "3")
        tasks, total, completed = translation.make_plan(args, self.paths)
        self.assertEqual((total, completed), (3, 0))
        self.assertEqual([task["count"] for task in tasks], [2, 1])
        self.assertEqual(translation.main(["--data-dir", str(self.train), "--output-dir", str(self.output),
                                           "--dry-run", "--limit", "3"]), 0)
        self.assertFalse(self.output.exists())

    def test_resume_after_crash_and_torn_write_skips_saved_samples(self):
        self.prepare_output()
        args = self.args()
        task = {"source": str(self.paths[0]), "count": 2}
        seen = []

        class Interrupted:
            def translate(self, rows, attempt):
                seen.extend(row["sample_id"] for row in rows)
                if len(seen) > 1:
                    raise RuntimeError("simulated inference crash")
                return [("Add an object.", "stop") for row in rows]

        with self.assertRaisesRegex(RuntimeError, "simulated inference crash"):
            translation.process_shard(args, task, Interrupted())
        checkpoint = translation.checkpoint_path(self.output, self.paths[0])
        with checkpoint.open("ab") as handle:
            handle.write(b'{"sample_id": 2, "instruction_en": "torn')
        self.assertEqual(len(translation.read_checkpoint(checkpoint, translation.selected_rows(self.paths[0]))), 1)
        resumed = []

        class Resumed:
            def translate(self, rows, attempt):
                resumed.extend(row["sample_id"] for row in rows)
                return [("Change the red object to blue.", "stop") for row in rows]

        self.assertEqual(translation.process_shard(args, task, Resumed()), 0)
        self.assertEqual(resumed, [2])
        records = translation.read_checkpoint(checkpoint, translation.selected_rows(self.paths[0]))
        self.assertEqual(len(records), 2)
        self.assertEqual(len(checkpoint.read_text().splitlines()), 2)

    def test_invalid_responses_are_retried_without_redoing_successes(self):
        self.prepare_output()
        args = self.args("--batch-size", "2", "--retries", "1")
        seen = []

        class Translator:
            def translate(self, rows, attempt):
                seen.append([row["sample_id"] for row in rows])
                if attempt == 0:
                    return [("Add an object.", "stop"), ("Change the red", "length")]
                return [("Change the red object to blue.", "stop")]

        task = {"source": str(self.paths[0]), "count": 2}
        self.assertEqual(translation.process_shard(args, task, Translator()), 0)
        self.assertEqual(seen, [[0, 2], [2]])

    def test_checkpoint_detects_wrong_source_duplicates_and_complete_corruption(self):
        self.prepare_output()
        rows = translation.selected_rows(self.paths[0])
        checkpoint = translation.checkpoint_path(self.output, self.paths[0])
        record = {**rows[0], "instruction_en": "Add an object."}
        checkpoint.write_text(json.dumps({**record, "instruction_zh": "错误的原文"}) + "\n")
        with self.assertRaisesRegex(ValueError, "does not match source"):
            translation.read_checkpoint(checkpoint, rows, repair=True)
        checkpoint.write_text((json.dumps(record) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            translation.read_checkpoint(checkpoint, rows, repair=True)
        checkpoint.write_text(json.dumps(record) + "\n{broken}\n")
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint"):
            translation.read_checkpoint(checkpoint, rows, repair=True)

    def test_gpu_partition_and_translation_validation(self):
        args = self.args("--num-gpus", "8", "--gpu-ids", "2,3,4,5,6,7,8,9", "--tp-size", "2")
        self.assertEqual(translation.gpu_groups(args), [["2", "3"], ["4", "5"], ["6", "7"], ["8", "9"]])
        args.gpu_ids = "GPU-one,GPU-two"
        args.num_gpus, args.tensor_parallel_size = 2, 1
        self.assertEqual(translation.gpu_groups(args), [["GPU-one"], ["GPU-two"]])
        args.num_gpus = 8
        with self.assertRaisesRegex(ValueError, "only 2 IDs"):
            translation.gpu_groups(args)
        self.assertEqual(translation.validate_translation('Write "你好" on the sign.'), 'Write "你好" on the sign.')
        for text in ("", "未翻译", "<think>reason</think>Add a cat.", "```Add a cat.```"):
            with self.assertRaises(ValueError):
                translation.validate_translation(text)

    def test_output_lock_and_changed_model_or_source_are_rejected(self):
        args = self.args()
        config = translation.run_config(args, self.train, self.paths)
        with translation.output_lock(self.output):
            translation.check_config(self.output, config, write=True)
            with self.assertRaises(RuntimeError):
                with translation.output_lock(self.output):
                    pass
        args.model = "another-model"
        with self.assertRaisesRegex(ValueError, "model"):
            translation.check_config(self.output, translation.run_config(args, self.train, self.paths))
        args.model = config["model"]
        with self.paths[0].open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "inputs"):
            translation.check_config(self.output, translation.run_config(args, self.train, self.paths))

    def test_spawned_vllm_replicas_failure_resume_and_deterministic_merge(self):
        fake_dir = self.root / "fake_modules"
        fake_dir.mkdir()
        (fake_dir / "vllm.py").write_text(FAKE_VLLM)
        events_path = self.root / "events.jsonl"
        env = {**os.environ, "PYTHONPATH": str(fake_dir), "FAKE_VLLM_EVENTS": str(events_path),
               "FAKE_FAIL_PROMPT": "添加物体 0"}
        base_command = [sys.executable, str(Path(translation.__file__)),
                        "--data-dir", str(self.data), "--output-dir", str(self.output),
                        "--batch-size", "2", "--retries", "1"]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths}
        result = subprocess.run(base_command + ["--num-gpus", "2", "--gpu-ids", "3,5"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse((self.output / "translations.jsonl").exists())
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["pending"], 1)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        self.assertEqual({tuple(event["devices"]) for event in events if event["type"] == "init"}, {("3",), ("5",)})
        first_call_count = len(events)
        del env["FAKE_FAIL_PROMPT"]
        result = subprocess.run(base_command + ["--num-gpus", "1", "--gpu-ids", "5"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        resume_events = events[first_call_count:]
        self.assertEqual([event["prompts"] for event in resume_events if event["type"] == "generate"], [["添加物体 0"]])
        merged = self.output / "translations.jsonl"
        records = [json.loads(line) for line in merged.read_text().splitlines()]
        self.assertEqual([row["sample_id"] for row in records], [0, 2, 3, 5, 6, 8, 9, 11])
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["complete"], True)
        merged_bytes = merged.read_bytes()
        # A complete resume works even with no model dependency available.
        no_model_env = {**os.environ, "PYTHONPATH": str(self.root / "empty_modules")}
        result = subprocess.run(base_command + ["--num-gpus", "8"],
                                env=no_model_env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(merged.read_bytes(), merged_bytes)
        self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths}, before)

    def test_eight_gpu_replicas_and_eight_way_tensor_parallel(self):
        # Eight independent jobs exercise isolation for all eight physical GPU IDs.
        for shard in range(4, 8):
            with gzip.open(self.metadata / f"train-{shard:05d}.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"sample_id": shard * 3, "instruction": f"添加物体 {shard * 3}"},
                                        ensure_ascii=False) + "\n")
        (self.train / "manifest.json").write_text(json.dumps({"num_sample_shards": 8}))
        fake_dir = self.root / "fake_modules"
        fake_dir.mkdir()
        (fake_dir / "vllm.py").write_text(FAKE_VLLM)
        events_path = self.root / "events.jsonl"
        env = {**os.environ, "PYTHONPATH": str(fake_dir), "FAKE_VLLM_EVENTS": str(events_path)}
        command = [sys.executable, str(Path(translation.__file__)), "--data-dir", str(self.data),
                   "--num-gpus", "8", "--gpu-ids", "0,1,2,3,4,5,6,7"]
        result = subprocess.run(command + ["--output-dir", str(self.output)],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        self.assertEqual({tuple(event["devices"]) for event in events if event["type"] == "init"},
                         {(str(index),) for index in range(8)})
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["translated"], 12)
        events_path.write_text("")
        result = subprocess.run(command + ["--output-dir", str(self.root / "tp8"), "--tp-size", "8"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        inits = [event for event in events if event["type"] == "init"]
        self.assertEqual(len(inits), 1)
        self.assertEqual(inits[0]["devices"], [str(index) for index in range(8)])

    def test_sigterm_cleans_up_worker_and_preserves_durable_batches(self):
        fake_dir = self.root / "fake_modules"
        fake_dir.mkdir()
        (fake_dir / "vllm.py").write_text(FAKE_VLLM)
        env = {**os.environ, "PYTHONPATH": str(fake_dir), "FAKE_VLLM_EVENTS": str(self.root / "events.jsonl"),
               "FAKE_VLLM_DELAY": "1"}
        command = [sys.executable, str(Path(translation.__file__)), "--data-dir", str(self.data),
                   "--output-dir", str(self.output), "--num-gpus", "1", "--gpu-ids", "0", "--batch-size", "1"]
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        checkpoint = translation.checkpoint_path(self.output, self.paths[0])
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if checkpoint.exists() and checkpoint.stat().st_size > 0:
                    break
                if process.poll() is not None:
                    self.fail(process.communicate()[1])
                time.sleep(0.05)
            self.assertTrue(checkpoint.exists() and checkpoint.stat().st_size > 0)
            process.terminate()
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130, stderr)
            saved = translation.read_checkpoint(checkpoint, translation.selected_rows(self.paths[0]))
            self.assertGreaterEqual(len(saved), 1)
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
