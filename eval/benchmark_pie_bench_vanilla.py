"""PIE-Bench with the original mask-free EditMGT inference settings."""
from eval.benchmark_pie_bench import main, parse_args


if __name__ == "__main__":
    raise SystemExit(main(parse_args(native_editmgt=True)))
