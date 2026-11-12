from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .io_utils import read_json


def write_timeline(scenes_path: Path, output_csv: Path, image_dir: Path) -> None:
    doc = read_json(scenes_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["index", "id", "start", "end", "duration", "narration", "image_file", "image_exists", "visual_brief", "image_prompt"]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, scene in enumerate(doc.get("scenes", []), start=1):
            row = {key: scene.get(key, "") for key in fields}
            row["index"] = index
            row["image_exists"] = (image_dir / scene.get("image_file", "")).exists()
            writer.writerow(row)
    print(f"Wrote timeline: {output_csv}")


def make_contact_sheet(scenes_path: Path, image_dir: Path, output_path: Path, columns: int = 3) -> None:
    doc = read_json(scenes_path)
    scenes = doc.get("scenes", [])
    cell_w, cell_h, label_h = 480, 270, 58
    margin, gap = 18, 16
    rows = (len(scenes) + columns - 1) // columns
    canvas = Image.new("RGB", (margin * 2 + columns * cell_w + (columns - 1) * gap,
                                margin * 2 + rows * (cell_h + label_h) + max(0, rows - 1) * gap), "#f1eee7")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for i, scene in enumerate(scenes):
        col, row = i % columns, i // columns
        x = margin + col * (cell_w + gap)
        y = margin + row * (cell_h + label_h + gap)
        path = image_dir / scene.get("image_file", "")
        if path.exists():
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((cell_w, cell_h))
                    frame = Image.new("RGB", (cell_w, cell_h), "white")
                    frame.paste(image, ((cell_w - image.width) // 2, (cell_h - image.height) // 2))
                    canvas.paste(frame, (x, y))
            except Exception:
                draw.rectangle((x, y, x + cell_w, y + cell_h), fill="#d6c8bc")
                draw.text((x + 12, y + 12), "Unreadable image", fill="#422", font=font)
        else:
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill="#dedbd5")
            draw.text((x + 12, y + 12), "IMAGE NOT GENERATED", fill="#554", font=font)
        label = f"{scene.get('id')}  {scene.get('start', 0):.1f}s–{scene.get('end', 0):.1f}s"
        draw.text((x, y + cell_h + 7), label, fill="#171717", font=font)
        narration = str(scene.get("narration", ""))
        if len(narration) > 64:
            narration = narration[:61] + "..."
        draw.text((x, y + cell_h + 30), narration, fill="#48433e", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=88)
    print(f"Saved contact sheet: {output_path}")
