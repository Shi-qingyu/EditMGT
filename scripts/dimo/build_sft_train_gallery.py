"""Build a browser for separate-file SFT visualizations on Inter-Edit train."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    selection = manifest.get("selection", "unspecified")
    checkpoint = manifest.get("checkpoint")
    if checkpoint is None and (root / "run_config.json").is_file():
        checkpoint = json.loads((root / "run_config.json").read_text(encoding="utf-8"))["checkpoint"]
    if checkpoint is None:
        checkpoint = str(PROJECT_ROOT / "runs/inter_brush_maskedit_en_no_dilation_mrcache0.8_d0.1_fresh/checkpoint-6800")
    checkpoint_label = Path(checkpoint).name
    cards = []
    completed = 0
    for row in samples:
        output = root / row["name"] / "sft18.png"
        if not output.is_file():
            continue
        completed += 1
        with Image.open(root / row["mask"]) as mask:
            mask_ratio = sum(mask.convert("L").histogram()[128:]) / (mask.width * mask.height)
        figures = []
        for label, relative in (
            ("Source", row["source"]),
            ("Mask", row["mask"]),
            ("SFT 18 steps", f'{row["name"]}/sft18.png'),
            ("Target", row["target"]),
        ):
            uri = html.escape(relative, quote=True)
            figures.append(
                f'<figure><a href="{uri}"><img loading="lazy" src="{uri}" '
                f'alt="{html.escape(label, quote=True)}"></a><figcaption>{label}</figcaption></figure>'
            )
        cards.append(
            f'<section data-type="{html.escape(row["edit_type"], quote=True)}">'
            f'<h2>{html.escape(row["name"])} · {html.escape(row["edit_type"])}</h2>'
            f'<p>{html.escape(row["instruction"])}</p>'
            + (f'<p><strong>Selection note:</strong> {html.escape(row["selection_note"])}</p>'
               if row.get("selection_note") else "")
            + f'<p>Mask area: {mask_ratio:.1%} · Seed: {row["seed"]}</p>'
              f'<div class="images">{"".join(figures)}</div></section>'
        )
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CHECKPOINT_LABEL · Inter-Edit train</title>
<style>body{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;color:#202530}
h1{font-size:28px}h2{font-size:18px;margin-bottom:4px}p{margin:5px 0}nav{position:sticky;top:0;background:white;padding:12px 0;border-bottom:1px solid #ddd;z-index:2}
select{font:inherit;padding:5px}section{padding:20px 0;border-bottom:1px solid #ddd}.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:12px}
figure{margin:0}img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#eee;border:1px solid #ddd}figcaption{text-align:center;margin-top:4px}
@media(max-width:800px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<h1>CHECKPOINT_LABEL · Inter-Edit train visualizations</h1>
<p>Non-text edits only. Source, mask, SFT output and target are separate PNG files. These are training examples for paper demo selection, not held-out performance evidence.</p>
<p><a href="manifest.json">Sample manifest</a> · <a href="README.md">Run details</a></p>
<nav><label>Category <select id="category"><option value="all">All</option><option>Add</option><option>Local</option><option>Remove</option><option>Texture</option></select></label></nav>
''' + "\n".join(cards) + '''<script>document.getElementById('category').addEventListener('change',e=>{for(const card of document.querySelectorAll('section[data-type]'))card.hidden=e.target.value!=='all'&&card.dataset.type!==e.target.value});</script></html>'''
    page = page.replace("CHECKPOINT_LABEL", html.escape(checkpoint_label))
    (root / "index.html").write_text(page, encoding="utf-8")
    links = ["[all completed samples](index.html)", "[metadata](manifest.json)"]
    if (root / "review_scores.json").is_file():
        links.append("[heuristic review scores](review_scores.json)")
    if (root / "paper_candidates" / "index.html").is_file():
        links.append("[24 manually reviewed paper candidates](paper_candidates/index.html)")
    if (root / "selection.json").is_file():
        links.append("[manual selection record](selection.json)")
    if (root / "run_config.json").is_file():
        links.append("[run configuration](run_config.json)")
    (root / "README.md").write_text(
        f"# {checkpoint_label}: Inter-Edit train visualizations\n\n"
        f"Completed {completed}/{len(samples)} samples ({manifest['per_type']} selected per category). "
        f"All selected samples are English, non-text edits from the Inter-Edit training split. "
        f"Selection: {selection}; seed {manifest['seed']}.\n\n"
        f"Model: `{checkpoint}`. "
        f"Inference: 18 steps, CFG 6, BF16, provided mask, 1024 resolution. "
        f"Each sample directory stores separate `source.png`, `mask.png`, `target.png`, "
        f"`sft18.png`, and `sft18.json`.\n\n"
        f"Browse {', '.join(links)}. Review scores use target LPIPS gain and source preservation "
        f"only to order human inspection; they are not benchmark metrics. "
        f"Training examples are suitable for qualitative demos, not generalization claims.\n",
        encoding="utf-8",
    )
    print(f"Gallery: {completed}/{len(samples)} completed", flush=True)


if __name__ == "__main__":
    main()
