"""Label training previews with the fixed probe's source and instruction."""
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def annotated_preview(source_path, edit_path, instruction, output_path, step):
    with Image.open(source_path) as image:
        source = image.convert('RGB')
    with Image.open(edit_path) as image:
        edit = image.convert('RGB')
    side = edit.width
    source = source.resize(edit.size, Image.Resampling.LANCZOS)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 24) if font_path.is_file() else ImageFont.load_default()
    title_font = ImageFont.truetype(str(font_path), 28) if font_path.is_file() else font
    margin = 24
    lines = textwrap.wrap(instruction, width=max(40, side // 13), break_long_words=True)
    caption_height = 90 + 34 * max(1, len(lines))
    canvas = Image.new('RGB', (side * 2 + margin * 3, edit.height + caption_height + margin * 2), '#f8f8f8')
    canvas.paste(source, (margin, 58))
    canvas.paste(edit, (side + margin * 2, 58))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), 'Source', font=title_font, fill='#111111')
    draw.text((side + margin * 2, 16), f'DiMO step {step}', font=title_font, fill='#111111')
    y = edit.height + 72
    draw.text((margin, y), 'Edit instruction:', font=font, fill='#111111')
    for line in lines:
        y += 34
        draw.text((margin, y), line, font=font, fill='#111111')
    canvas.save(output_path)
