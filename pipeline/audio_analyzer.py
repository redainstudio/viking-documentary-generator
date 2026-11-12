from __future__ import annotations

import json
import re
import time
from pathlib import Path
from mutagen import File as MutagenFile
from typing import Any

from .io_utils import write_json_atomic


def _to_plain(value: Any) -> Any:
    """Convert SDK model objects to JSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_plain(value.model_dump(exclude_none=True))
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _to_plain(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*s?", text)
    if match:
        return float(match.group(1))
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def _extract_word_annotations(interaction: Any) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    steps = getattr(interaction, "steps", None) or []
    for step in steps:
        content_items = getattr(step, "content", None) or []
        for content in content_items:
            annotations = getattr(content, "annotations", None) or []
            for annotation in annotations:
                kind = getattr(annotation, "type", None)
                if kind is None and isinstance(annotation, dict):
                    kind = annotation.get("type")
                if kind != "word_info":
                    continue
                raw = _to_plain(annotation)
                if not isinstance(raw, dict):
                    continue
                word = raw.get("text") or raw.get("word") or raw.get("content")
                start = _seconds(_first_present(raw, "start_offset", "start_time", "start"))
                end = _seconds(_first_present(raw, "end_offset", "end_time", "end"))
                if word and start is not None and end is not None and end >= start:
                    words.append({"text": str(word), "start": start, "end": end})
    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def _extract_annotations_fallback(interaction: Any) -> list[dict[str, Any]]:
    """Handle SDK versions that expose response payload as dictionaries."""
    raw = _to_plain(interaction)
    words: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "word_info":
                word = node.get("text") or node.get("word") or node.get("content")
                start = _seconds(_first_present(node, "start_offset", "start_time", "start"))
                end = _seconds(_first_present(node, "end_offset", "end_time", "end"))
                if word and start is not None and end is not None and end >= start:
                    words.append({"text": str(word), "start": start, "end": end})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(raw)
    unique = {(w["start"], w["end"], w["text"]): w for w in words}
    return sorted(unique.values(), key=lambda item: (item["start"], item["end"]))


def transcribe_audio(audio_path: Path, output_path: Path, model: str, api_key: str) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise ValueError("Audio file is empty.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Missing dependency google-genai. Install requirements.txt first.") from exc

    client = genai.Client(api_key=api_key)
    print(f"Uploading {audio_path.name} to Gemini Files API...")
    uploaded = client.files.upload(file=str(audio_path))
    print(f"Transcribing with {model} (verbatim + word timestamps)...")
    interaction = client.interactions.create(
        model=model,
        input=[{"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type}],
        generation_config={
            "transcription_config": {
                "mode": {"type": "verbatim", "timestamp_granularities": ["word"]}
            }
        },
    )

    transcript = getattr(interaction, "output_text", None) or ""
    word_items = _extract_word_annotations(interaction)
    if not word_items:
        word_items = _extract_annotations_fallback(interaction)
    if not transcript and word_items:
        transcript = " ".join(item["text"] for item in word_items)
    if not transcript:
        raise RuntimeError("Gemini returned no transcript text. Check model access and the audio file.")
    if not word_items:
        raise RuntimeError(
            "Gemini returned transcript text but no word-level annotations. "
            "The scene planner needs timestamps; inspect SDK/API compatibility before proceeding."
        )

    previous_end = -1.0
    for index, item in enumerate(word_items):
        if item["start"] < 0 or item["end"] < item["start"]:
            raise ValueError(f"Invalid word timestamp at index {index}: {item}")
        if item["start"] + 0.05 < previous_end:
            print(f"Warning: overlapping word timestamps near {item['text']!r}.")
        previous_end = max(previous_end, item["end"])

    media = MutagenFile(str(audio_path))
    audio_duration = float(media.info.length) if media and getattr(media, "info", None) else max(item["end"] for item in word_items)
    payload = {
        "source_audio": audio_path.name,
        "model": model,
        "transcript": transcript,
        "words": word_items,
        "duration_seconds": audio_duration,
        "last_word_end_seconds": max(item["end"] for item in word_items),
        "word_count": len(word_items),
        "created_at_unix": time.time(),
    }
    write_json_atomic(output_path, payload)
    print(f"Saved {len(word_items)} word timestamps to {output_path}")
    print(f"Timed transcript duration: {payload['duration_seconds']:.2f}s")
    return payload
