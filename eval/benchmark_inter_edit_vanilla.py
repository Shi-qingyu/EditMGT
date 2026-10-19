"""Benchmark original mask-free EditMGT on the English Inter-Edit test split."""

from eval.benchmark_inter_edit import main, parse_args


if __name__ == "__main__":
    raise SystemExit(main(parse_args(native_editmgt=True)))
