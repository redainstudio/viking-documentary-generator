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


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    model: str,
    api_key: str,
    max_scene_seconds: float = 4.8,
) -> dict[str, Any]:
    """Transcribe audio with Gemini Flash audio understanding.

    Gemini Flash is a general audio-understanding model, not a dedicated forced
    aligner. It returns timestamped narration segments; timestamps are model
    estimates and should be checked against the audio before final editing.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise ValueError("Audio file is empty.")
    if not 0 < max_scene_seconds <= 5:
        raise ValueError("Maximum scene duration must be greater than 0 and no more than 5 seconds.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Missing dependency google-genai. Install requirements.txt first.") from exc

    media = MutagenFile(str(audio_path))
    if not media or not getattr(media, "info", None):
        raise RuntimeError("Could not read audio duration. Verify the MP3/WAV file is valid.")
    audio_duration = float(media.info.length)
    if audio_duration <= 0:
        raise RuntimeError("Audio duration is invalid.")

    client = genai.Client(api_key=api_key)
    print(f"Uploading {audio_path.name} to Gemini Files API...")
    uploaded = client.files.upload(file=str(audio_path))
    prompt = f"""
Transcribe the supplied narration accurately and preserve the spoken wording.
The audio duration is approximately {audio_duration:.2f} seconds.

Return ONLY valid JSON with this exact structure:
{{
  "transcript": "the complete transcript in normal readable form",
  "segments": [
    {{"start": 0.0, "end": 4.7, "text": "the exact words spoken in this interval"}}
  ]
}}

Timing requirements:
- Segment timestamps are absolute seconds from the beginning of the audio.
- Cover the narration from the beginning to the end in chronological order.
- Each segment must be no longer than {max_scene_seconds:.2f} seconds.
- Keep each segment's text in the exact spoken order. Do not summarize, paraphrase, omit, or invent words.
- Place sentence boundaries at natural pauses where possible.
- Include brief silence as a segment only when it is visually useful; otherwise adjacent segments may have a small gap.
- Use decimal seconds, not timecode strings.
- Do not include markdown fences, commentary, or any keys other than transcript and segments.
""".strip()

    print(f"Transcribing with {model} using Gemini audio understanding...")
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "text", "text": prompt},
            {"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
        ],
    )
    response_text = getattr(interaction, "output_text", None) or ""
    if not response_text.strip():
        raise RuntimeError("Gemini returned an empty audio transcription response.")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^\`\`\`(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s*\`\`\`$", "", cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini did not return the required JSON transcript with timed segments. "
            f"Response preview: {response_text[:700]}"
        ) from exc

    transcript = str(result.get("transcript", "")).strip()
    raw_segments = result.get("segments") or []
    if not transcript or not isinstance(raw_segments, list) or not raw_segments:
        raise RuntimeError("Gemini response is missing transcript text or timed segments.")

    segments: list[dict[str, Any]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item["start"]))
            end = min(audio_duration, float(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        if text and end > start:
            segments.append({"start": start, "end": end, "text": text})

    segments.sort(key=lambda item: (item["start"], item["end"]))
    if not segments:
        raise RuntimeError("Gemini returned no usable timestamped narration segments.")

    # Enforce the hard five-second visual limit even if the model returns a
    # slightly longer segment. Text is divided in order across equal time slices.
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        duration = segment["end"] - segment["start"]
        part_count = max(1, int((duration + max_scene_seconds - 1e-6) // max_scene_seconds))
        words = segment["text"].split()
        if part_count == 1:
            normalized.append(segment)
            continue
        for part_index in range(part_count):
            left = round(len(words) * part_index / part_count)
            right = round(len(words) * (part_index + 1) / part_count)
            start = segment["start"] + duration * part_index / part_count
            end = segment["start"] + duration * (part_index + 1) / part_count
            text = " ".join(words[left:right]).strip()
            if text and end > start:
                normalized.append({"start": start, "end": end, "text": text})

    normalized.sort(key=lambda item: (item["start"], item["end"]))
    payload = {
        "source_audio": audio_path.name,
        "model": model,
        "transcript": transcript,
        "segments": normalized,
        "words": [],
        "duration_seconds": audio_duration,
        "last_segment_end_seconds": max(item["end"] for item in normalized),
        "word_count": len(transcript.split()),
        "segment_count": len(normalized),
        "timestamp_note": "Segment timestamps are Gemini Flash audio-understanding estimates, not dedicated forced-alignment annotations.",
        "created_at_unix": time.time(),
    }
    write_json_atomic(output_path, payload)
    print(f"Saved {len(normalized)} timed narration segments to {output_path}")
    print(f"Audio duration: {audio_duration:.2f}s")
    return payload
