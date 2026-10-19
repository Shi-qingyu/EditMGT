"""Benchmark original mask-free EditMGT on MagicBrush dev; save edits and metrics."""

from eval.benchmark_magicbrush import main, parse_args


if __name__ == "__main__":
    raise SystemExit(main(parse_args(native_editmgt=True)))
