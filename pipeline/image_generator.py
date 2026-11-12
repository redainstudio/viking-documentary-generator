from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json_atomic

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}


def _load_references(reference_dir: Path) -> list[Any]:
    try:
        from PIL import Image
    except ImportError:
        return []
    references = []
    if reference_dir.exists():
        for path in sorted(reference_dir.iterdir()):
            if path.suffix.lower() in SUPPORTED_IMAGES:
                try:
                    references.append(Image.open(path).convert("RGB"))
                except Exception as exc:
                    print(f"Warning: could not load reference {path.name}: {exc}")
    return references[:4]


def _extract_image(response: Any) -> bytes | None:
    for part in getattr(response, "parts", []) or []:
        inline = getattr(part, "inline_data", None)
        data = getattr(inline, "data", None) if inline else None
        if data:
            return data.encode("utf-8") if isinstance(data, str) else data
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline else None
            if data:
                return data.encode("utf-8") if isinstance(data, str) else data
    return None


def generate_images(
    scenes_path: Path,
    output_dir: Path,
    reference_dir: Path,
    model: str,
    api_key: str,
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
    retries: int = 3,
    limit: int | None = None,
    overwrite: bool = False,
    scene_ids: set[str] | None = None,
) -> None:
    document = read_json(scenes_path)
    scenes: list[dict[str, Any]] = document.get("scenes", [])
    if not scenes:
        raise ValueError("Storyboard contains no scenes. Run the plan command first.")
    output_dir.mkdir(parents=True, exist_ok=True)
    references = _load_references(reference_dir)
    if references:
        print(f"Using {len(references)} visual reference image(s).")
    else:
        print("No visual reference images found. Text-only style guidance will be used.")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Missing dependency google-genai. Install requirements.txt first.") from exc
    client = genai.Client(api_key=api_key)
    eligible = [scene for scene in scenes if scene_ids is None or scene.get("id") in scene_ids]
    selected = eligible[:limit] if limit else eligible
    if not selected:
        raise ValueError("No scenes matched the requested selection.")
    for index, scene in enumerate(selected, start=1):
        image_path = output_dir / scene["image_file"]
        if image_path.exists() and not overwrite:
            print(f"[{index}/{len(selected)}] Skip existing {image_path.name}")
            scene["status"] = "generated"
            continue

        print(f"[{index}/{len(selected)}] Generating {scene['id']} ({scene['start']:.2f}-{scene['end']:.2f}s)")
        contents: list[Any] = [
            "Use attached images only as visual style and character-design references; do not copy their scene composition. "
            "Preserve the simple white round-headed stickman anatomy, bold black hand-drawn outlines and expressive minimal faces. "
            "Follow the scene-specific historical visual brief exactly. Generate one single coherent widescreen 16:9 illustration, "
            "not a collage or panel layout. No words, captions, labels, watermarks, UI or borders.\n\n" + scene["image_prompt"]
        ]
        contents.extend(references)
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        response_format={"image": {"aspect_ratio": aspect_ratio, "image_size": image_size}},
                    ),
                )
                image_bytes = _extract_image(response)
                if not image_bytes:
                    raise RuntimeError("Image model returned no image data. Check model capability and API access.")
                image_path.write_bytes(image_bytes)
                scene["status"] = "generated"
                scene["image_error"] = None
                print(f"    Saved {image_path}")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    delay = min(60, (2 ** (attempt - 1)) * 3 + random.random())
                    print(f"    Attempt {attempt} failed: {exc}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        if last_error:
            scene["status"] = "failed"
            scene["image_error"] = str(last_error)
            print(f"    FAILED {scene['id']}: {last_error}")
        document["scenes"] = scenes
        document["generated_count"] = sum(s.get("status") == "generated" for s in scenes)
        document["failed_count"] = sum(s.get("status") == "failed" for s in scenes)
        write_json_atomic(scenes_path, document)

    for scene in scenes:
        path = output_dir / scene["image_file"]
        if path.exists():
            scene["status"] = "generated"
    document["scenes"] = scenes
    document["generated_count"] = sum(s.get("status") == "generated" for s in scenes)
    document["failed_count"] = sum(s.get("status") == "failed" for s in scenes)
    write_json_atomic(scenes_path, document)
