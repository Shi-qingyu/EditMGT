"""Single-host, one-model-per-GPU MagicBrush evaluation.

The parent owns BenchmarkRun's directory lock for the entire launch. Spawned
workers only write disjoint sample files and their own metadata directories.
No process group or gradient synchronization is needed for inference.
"""
import copy
import json
import os
import time
import uuid
from pathlib import Path

import torch
import torch.multiprocessing as mp

from src.benchmark_data import BenchmarkImages
from src.benchmark_utils import (
    BenchmarkRun, atomic_json, build_metric_summary, generate_benchmark,
    score_benchmark, write_generation_summary,
)


class _BenchmarkShard(BenchmarkRun):
    """Worker view; the live parent already holds the output-directory lock."""

    def __init__(self, root, info, selection):
        self.root = Path(root)
        self.info = info
        self.selection = selection
        self.by_key = {task["key"]: task for task in selection}
        self._lock = None


def shard_magicbrush_sessions(selection, num_workers):
    """Keep every variant/turn of a session together, in original task order."""
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    counts = {}
    for task in selection:
        session_id = task["session_id"]
        counts[session_id] = counts.get(session_id, 0) + 1
    if not counts:
        raise ValueError("The selected MagicBrush subset is empty")
    num_workers = min(num_workers, len(counts))
    loads = [0] * num_workers
    owners = {}
    for session_id, count in counts.items():
        rank = min(range(num_workers), key=lambda index: loads[index])
        owners[session_id] = rank
        loads[rank] += count
    shards = [[] for _ in range(num_workers)]
    for task in selection:
        shards[owners[task["session_id"]]].append(task)
    for shard in shards:
        seen = set()
        for task in shard:
            parent = task.get("input_prediction_key")
            if parent is not None and parent not in seen:
                raise ValueError(f"Multi-turn dependency {parent} must precede {task['key']} on the same GPU")
            seen.add(task["key"])
    return shards


def _worker_devices(args, num_workers):
    device = torch.device(args.device)
    metric_device = torch.device(args.metric_device or args.device)
    if device.type != "cuda":
        raise ValueError("--num-gpus > 1 requires a CUDA --device; use --num-gpus 1 for CPU evaluation")
    if metric_device.type not in ("cuda", "cpu"):
        raise ValueError("Parallel scoring requires a CUDA or CPU --metric-device")
    start = device.index or 0
    metric_start = metric_device.index or 0
    devices = [f"cuda:{start + rank}" for rank in range(num_workers)]
    metric_devices = [f"cuda:{metric_start + rank}" if metric_device.type == "cuda" else "cpu"
                      for rank in range(num_workers)]
    used = (devices if args.stage != "metrics" else []) + (metric_devices if args.stage != "generate" else [])
    visible = torch.cuda.device_count()
    if any(torch.device(name).type == "cuda" and torch.device(name).index >= visible for name in used):
        raise ValueError(
            f"{num_workers} workers starting at --device {args.device} / "
            f"--metric-device {args.metric_device or args.device} exceed {visible} visible CUDA devices. "
            "Select GPUs with CUDA_VISIBLE_DEVICES or lower --num-gpus."
        )
    return devices, metric_devices


def _benchmark_worker(rank, args, root, info, shards, names, worker_dirs, devices, metric_devices):
    args = copy.copy(args)
    args.device = devices[rank]
    args.metric_device = metric_devices[rank]
    if args.stage != "metrics":
        torch.cuda.set_device(args.device)
    elif torch.device(args.metric_device).type == "cuda":
        torch.cuda.set_device(args.metric_device)
    run = _BenchmarkShard(root, info, shards[rank])
    images = BenchmarkImages()
    worker_dir = Path(worker_dirs[rank])
    worker_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage != "metrics":
            generate_benchmark(args, run, images, output_dir=worker_dir,
                               progress_desc=f"Generate {args.device}")
        if args.stage != "generate":
            score_benchmark(args, run, images, names, output_dir=worker_dir / "metrics",
                            model_lock=run.root / ".metric_models.lock",
                            progress_desc=f"Metrics {args.metric_device}", quiet=True)
    finally:
        images.close()


def _spawn_workers(args, run, shards, names, worker_dirs, devices, metric_devices):
    context = mp.spawn(
        _benchmark_worker,
        args=(args, run.root, run.info, shards, names, worker_dirs, devices, metric_devices),
        nprocs=len(shards), join=False,
    )
    try:
        while not context.join():
            pass
    except BaseException:
        # Also stop children on Ctrl-C, before the parent releases the global
        # output lock or reads partial results. spawn handles worker failures;
        # explicit cleanup covers interruption of its join loop as well.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        raise


def _read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            # A worker terminated by spawn's error handling may leave an
            # unfinished final write. Previously flushed records remain usable.
            if not line.endswith("\n"):
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _merge_metrics(run, names, worker_dirs, parallel_info):
    rows, failures, protocols = [], [], []
    for worker_dir in worker_dirs:
        directory = worker_dir / "metrics"
        rows.extend(_read_jsonl(directory / "per_sample.jsonl"))
        failures.extend(_read_jsonl(directory / "errors.jsonl"))
        if (directory / "protocol.json").is_file():
            protocols.append(json.loads((directory / "protocol.json").read_text(encoding="utf-8")))
    if protocols and any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("GPU workers used different metric protocols")
    order = {task["key"]: index for index, task in enumerate(run.selection)}
    keys = [row["key"] for row in rows + failures]
    if len(keys) != len(set(keys)) or set(keys) - set(order):
        raise ValueError("Parallel scoring produced duplicate or unexpected sample keys")
    rows.sort(key=lambda row: order[row["key"]])
    failures.sort(key=lambda row: order[row["key"]])
    records = [run.record(row["key"]) for row in rows]
    protocol = protocols[0] if protocols else {"requested": list(names), "initialization_incomplete": True}
    summary = build_metric_summary(run, rows, failures, records, protocol)
    summary["data_parallel"] = parallel_info
    summary["complete"] = summary["complete"] and parallel_info["workers_finished"]
    directory = run.root / "metrics"
    _atomic_jsonl(directory / "per_sample.jsonl", rows)
    _atomic_jsonl(directory / "errors.jsonl", failures)
    atomic_json(directory / "protocol.json", protocol)
    atomic_json(directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["complete"]


def run_magicbrush_data_parallel(args, run, names):
    """Launch GPU replicas and publish global summaries while holding run's lock."""
    shards = shard_magicbrush_sessions(run.selection, args.num_gpus)
    devices, metric_devices = _worker_devices(args, len(shards))
    # A distinct directory per invocation prevents stale partial scores from a
    # previous launch being merged after a failed or differently sized resume.
    launch_dir = run.root / "workers" / uuid.uuid4().hex
    worker_dirs = [launch_dir / f"rank-{rank:02d}" for rank in range(len(shards))]
    parallel_info = {
        "mode": "data_parallel", "requested_gpus": args.num_gpus, "world_size": len(shards),
        "partition": "complete_magicbrush_sessions", "stage": args.stage,
        "worker_outputs": str(launch_dir.relative_to(run.root)), "workers_finished": False, "phases": {},
        "workers": [
            {"rank": rank, "device": devices[rank], "metric_device": metric_devices[rank],
             "sessions": len({task["session_id"] for task in shard}), "predictions": len(shard)}
            for rank, shard in enumerate(shards)
        ],
    }
    atomic_json(run.root / "parallel_execution.json", parallel_info)
    print(f"Launching {len(shards)} MagicBrush workers on {', '.join(devices)}", flush=True)
    started = time.perf_counter()
    complete = False
    try:
        # Finish/release every editing replica before starting metric models.
        # This also supports a different CUDA starting device for scoring.
        stages = ("generate", "metrics") if args.stage == "all" else (args.stage,)
        for stage in stages:
            phase_args = copy.copy(args)
            phase_args.stage = stage
            phase_started = time.perf_counter()
            phase = {"workers_finished": False}
            parallel_info["phases"][stage] = phase
            try:
                _spawn_workers(phase_args, run, shards, names, worker_dirs, devices, metric_devices)
                phase["workers_finished"] = True
            finally:
                phase["wall_seconds"] = time.perf_counter() - phase_started
        parallel_info["workers_finished"] = True
    except BaseException as error:
        parallel_info["error_type"] = type(error).__name__
        parallel_info["error"] = str(error)
        raise
    finally:
        parallel_info["wall_seconds"] = time.perf_counter() - started
        for worker, worker_dir in zip(parallel_info["workers"], worker_dirs):
            path = worker_dir / "execution.json"
            if path.is_file():
                worker["execution"] = json.loads(path.read_text(encoding="utf-8"))
        atomic_json(run.root / "parallel_execution.json", parallel_info)
        if args.stage != "metrics":
            # Records/images are shared and uniquely owned. Read all samples
            # here instead of averaging worker summaries of different sizes.
            generation = write_generation_summary(run)
            generation["complete"] = (
                generation["complete"]
                and parallel_info["phases"].get("generate", {}).get("workers_finished", False)
            )
            generation["data_parallel"] = parallel_info
            atomic_json(run.root / "generation_summary.json", generation)
            atomic_json(run.root / "execution.json", parallel_info)
            complete = generation["complete"] and parallel_info["workers_finished"]
        if args.stage != "generate":
            complete = _merge_metrics(run, names, worker_dirs, parallel_info)
    return complete
