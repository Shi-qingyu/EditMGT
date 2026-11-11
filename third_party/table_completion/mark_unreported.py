"""Mark genuinely unavailable table cells without inventing benchmark scores."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CELL = re.compile(r"(?<=& )--(?=\s*(?:&|\\\\))")


def main():
    for name in ("magicbrush.tex", "pie_inter.tex"):
        path = ROOT / "docs" / name
        before = path.read_text(encoding="utf-8")
        after, count = CELL.subn("N/R", before)
        if CELL.search(after):
            raise AssertionError(f"Unmarked data cell in {path}")
        path.write_text(after, encoding="utf-8")
        print(f"{name}: marked {count} unavailable cells")


if __name__ == "__main__":
    main()
