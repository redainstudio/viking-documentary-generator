from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json_atomic

STYLE_BIBLE = """
VISUAL STYLE — MUST BE FOLLOWED IN EVERY IMAGE:
A clean, hand-drawn 2D stickman documentary illustration, matching the supplied reference artwork. Human characters are simple white circular heads with expressive minimal black facial features, slim black stick-like arms and legs, bold slightly imperfect black ink outlines, and simple readable body shapes. Historical characters may wear scene-appropriate clothing, helmets, tunics, cloaks, armour and accessories, but retain the same recognizable simple stickman anatomy and white round heads. Environments are not empty: draw richly composed, historically grounded locations with boats, timber buildings, landscapes, weather, furniture, trees, tools and period props as appropriate. Use a restrained cinematic palette appropriate to the scene: cold blue-grey fog for ominous sea scenes, earthy browns and muted greens for daily life, warm firelight for interiors, and dramatic but legible contrast for battles and mythology. Hand-drawn storybook / explainer-documentary look, not photorealistic, not 3D, not anime, not glossy vector clip-art. Strong clear silhouettes, readable action, balanced 16:9 composition, carefully controlled detail. The characters must remain visibly stick figures, not realistic human bodies.

CONSISTENCY RULES:
- Use the same visual grammar, head shape, face style, line weight and body proportions in every scene.
- Recurring people should retain recognizable clothing colours and accessories across adjacent scenes when the narration identifies the same person.
- For historical Viking scenes, use plausible Viking Age material culture. Do NOT use horned helmets as ordinary Viking equipment. Do not insert fantasy horns, modern objects, modern buildings, firearms, text labels, captions, logos or watermarks unless specifically required by the narration.
- Depict mythology as a deliberate mythological illustration, not as a claim that supernatural beings are historically real.
- Never invent written text in the image. No speech bubbles, subtitles, UI, borders or collages.
- One coherent scene per image. Do not split the frame into panels.
""".strip()

PLANNER_SYSTEM = """
You are an expert historical documentary storyboard artist and prompt engineer.
Create exactly one image brief for each supplied narration window. The windows already have accurate absolute start/end times. Preserve their order and timing exactly. Never add, remove, merge, split or reorder a window. The visual should directly illustrate the words spoken during that interval, not a generic image about the overall topic.

For every window:
- Write a concise but specific visual description of what must be visibly happening.
- Name the main subject(s), physical action, location/environment, period-appropriate objects/clothing, camera framing and lighting/mood.
- Use different visual compositions across consecutive scenes. Vary wide establishing views, medium action views, close-ups, maps rendered as physical parchment (without text), object details and character reactions.
- Prefer literal visual evidence for historical facts. For uncertain claims or legendary material, frame it as a saga, belief, reconstruction or theory rather than asserting it as established fact.
- For narration that asks a question, use a visually intriguing but relevant scene; do not put written question text in the image.
- Keep each prompt self-contained. Do not refer to a previous image or say "same as before".
- Do not add captions, readable lettering or watermarks.

Return valid JSON only, matching the requested schema. Every input id must appear exactly once.
""".strip()

SCENE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "visual_brief": {"type": "STRING"},
                    "composition": {"type": "STRING"},
                    "historical_notes": {"type": "STRING"},
                    "mood": {"type": "STRING"},
                },
                "required": ["id", "visual_brief", "composition", "historical_notes", "mood"],
            },
        }
    },
    "required": ["scenes"],
}


def _sentence_end(word: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]*$", word.strip()))


def build_timed_windows(
    words: list[dict[str, Any]], max_seconds: float = 4.8, min_seconds: float = 2.0,
    audio_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Create continuous, sentence-aware windows. Every window is <= max_seconds."""
    if not words:
        raise ValueError("Transcript contains no timestamped words.")
    if max_seconds <= 0 or max_seconds > 5:
        raise ValueError("Maximum scene duration must be > 0 and <= 5 seconds.")

    normalized = sorted(words, key=lambda w: (float(w["start"]), float(w["end"])))
    duration_total = float(audio_duration or max(float(w["end"]) for w in normalized))
    if duration_total <= 0:
        raise ValueError("Audio duration must be positive.")
    windows: list[dict[str, Any]] = []
    index = 0
    cursor = 0.0

    while cursor < duration_total - 0.01:
        hard_end = min(duration_total, cursor + max_seconds)
        selected: list[dict[str, Any]] = []
        while index < len(normalized):
            word = normalized[index]
            w_start, w_end = float(word["start"]), float(word["end"])
            if w_end < cursor - 0.05:
                index += 1
                continue
            if selected and _sentence_end(str(selected[-1]["text"])):
                if float(selected[-1]["end"]) - cursor >= min_seconds:
                    break
            if w_start >= hard_end:
                break
            if w_end > hard_end + 0.02 and selected:
                break
            selected.append(word)
            index += 1

        segment_text = " ".join(str(w["text"]) for w in selected).strip()
        elapsed = hard_end - cursor
        if selected and elapsed >= min_seconds:
            last = selected[-1]
            next_word = normalized[index] if index < len(normalized) else None
            if _sentence_end(str(last["text"])) and next_word:
                gap_midpoint = (float(last["end"]) + float(next_word["start"])) / 2
                candidate = max(float(last["end"]), gap_midpoint)
                if cursor + min_seconds <= candidate <= hard_end:
                    hard_end = candidate
                    elapsed = hard_end - cursor

        if hard_end <= cursor + 0.001:
            hard_end = min(duration_total, cursor + max_seconds)

        if selected:
            segment_text = " ".join(str(w["text"]) for w in selected).strip()
            word_count = len(selected)
        else:
            next_words = normalized[index : min(index + 12, len(normalized))]
            context = " ".join(str(w["text"]) for w in next_words).strip()
            segment_text = f"[Brief narration pause; transition toward: {context}]" if context else "[Brief narration pause; hold the closing visual]"
            word_count = 0

        windows.append({
            "id": f"scene_{len(windows) + 1:04d}",
            "start": round(cursor, 3),
            "end": round(hard_end, 3),
            "duration": round(hard_end - cursor, 3),
            "narration": segment_text,
            "word_count": word_count,
            "is_pause": word_count == 0,
        })
        cursor = hard_end

    if index < len(normalized):
        trailing = normalized[index:]
        if windows:
            windows[-1]["narration"] += " " + " ".join(str(w["text"]) for w in trailing)
            windows[-1]["word_count"] += len(trailing)

    for idx, window in enumerate(windows, start=1):
        window["id"] = f"scene_{idx:04d}"
        if window["duration"] > max_seconds + 0.001:
            raise ValueError(f"Generated window exceeds configured maximum: {window}")
    return windows


def _build_windows_from_segments(
    segments: list[dict[str, Any]], audio_duration: float, max_seconds: float
) -> list[dict[str, Any]]:
    """Convert Gemini's timed narration segments into fixed, ordered scene windows."""
    if not segments:
        raise ValueError("Transcript contains no timed narration segments.")
    if not 0 < max_seconds <= 5:
        raise ValueError("Maximum scene duration must be > 0 and <= 5 seconds.")

    normalized = sorted(segments, key=lambda item: (float(item["start"]), float(item["end"])))
    windows: list[dict[str, Any]] = []
    cursor = 0.0

    def append_window(start: float, end: float, narration: str, is_pause: bool = False) -> None:
        if end <= start + 0.001:
            return
        windows.append({
            "id": f"scene_{len(windows) + 1:04d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "narration": narration,
            "word_count": 0 if is_pause else len(narration.split()),
            "is_pause": is_pause,
        })

    def append_pause(start: float, end: float) -> None:
        while start < end - 0.01:
            next_end = min(end, start + max_seconds)
            append_window(start, next_end, "[Brief narration pause; hold a relevant transition visual]", True)
            start = next_end

    for segment in normalized:
        start = max(cursor, max(0.0, float(segment["start"])))
        end = min(audio_duration, float(segment["end"]))
        narration = str(segment.get("text", "")).strip()
        if not narration or end <= start + 0.001:
            continue
        if start - cursor > 0.08:
            append_pause(cursor, start)

        duration = end - start
        part_count = max(1, int((duration + max_seconds - 1e-6) // max_seconds))
        words = narration.split()
        for part_index in range(part_count):
            left = round(len(words) * part_index / part_count)
            right = round(len(words) * (part_index + 1) / part_count)
            part_start = start + duration * part_index / part_count
            part_end = start + duration * (part_index + 1) / part_count
            part_text = " ".join(words[left:right]).strip()
            if part_text:
                append_window(part_start, part_end, part_text)
        cursor = max(cursor, end)

    if audio_duration - cursor > 0.08:
        append_pause(cursor, audio_duration)

    for index, window in enumerate(windows, start=1):
        window["id"] = f"scene_{index:04d}"
        if window["duration"] > max_seconds + 0.001:
            raise ValueError(f"Timed segment exceeds maximum scene duration: {window}")
    if not windows:
        raise ValueError("No usable scene windows could be built from the timed segments.")
    return windows


def _planner_call(client: Any, model: str, windows: list[dict[str, Any]]) -> dict[str, Any]:
    from google.genai import types
    prompt = {
        "task": "Create one specific documentary image brief per timed narration window.",
        "windows": windows,
        "output_requirements": {
            "one_scene_per_window": True,
            "preserve_ids_exactly": True,
            "no_timing_changes": True,
            "no_text_in_images": True,
        },
    }
    response = client.models.generate_content(
        model=model,
        contents=[PLANNER_SYSTEM, json.dumps(prompt, ensure_ascii=False)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCENE_SCHEMA,
            temperature=0.35,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini storyboard planner returned an empty response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Planner response was not valid JSON: {text[:500]}") from exc


def plan_scenes(
    transcript_path: Path,
    output_path: Path,
    model: str,
    api_key: str,
    max_seconds: float,
    min_seconds: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    transcript = read_json(transcript_path)
    audio_duration = float(transcript.get("duration_seconds") or 0)
    segments = transcript.get("segments") or []
    words = transcript.get("words") or []
    # Prefer word-level timestamps whenever available. Whisper's segment
    # boundaries can span several seconds, so splitting segment text evenly
    # would only estimate word positions and weaken scene-to-audio alignment.
    if words:
        windows = build_timed_windows(
            words,
            max_seconds=max_seconds,
            min_seconds=min_seconds,
            audio_duration=audio_duration or max(float(w["end"]) for w in words),
        )
    elif segments:
        windows = _build_windows_from_segments(segments, audio_duration, max_seconds)
    else:
        raise ValueError("Transcript has neither timed segments nor word-level timestamps.")
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Missing dependency google-genai. Install requirements.txt first.") from exc
    client = genai.Client(api_key=api_key)
    planned: list[dict[str, Any]] = []

    for offset in range(0, len(windows), batch_size):
        batch = windows[offset : offset + batch_size]
        contextual_batch = []
        for local_index, window in enumerate(batch):
            global_index = offset + local_index
            contextual_batch.append({
                **window,
                "context_before": windows[global_index - 1]["narration"] if global_index > 0 else "",
                "context_after": windows[global_index + 1]["narration"] if global_index + 1 < len(windows) else "",
            })
        print(f"Planning scenes {offset + 1}-{offset + len(batch)} of {len(windows)}...")
        result = _planner_call(client, model, contextual_batch)
        results = result.get("scenes", [])
        by_id = {str(item.get("id")): item for item in results if isinstance(item, dict)}
        expected_ids = [w["id"] for w in batch]
        if set(by_id) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(by_id))
            extra = sorted(set(by_id) - set(expected_ids))
            raise RuntimeError(f"Planner IDs mismatch. Missing={missing}; unexpected={extra}")
        for window in batch:
            details = by_id[window["id"]]
            brief = str(details.get("visual_brief", "")).strip()
            if len(brief) < 20:
                raise RuntimeError(f"Planner returned an inadequate visual brief for {window['id']}.")
            prompt = "\n\n".join([
                STYLE_BIBLE,
                "SCENE TO ILLUSTRATE:",
                brief,
                f"COMPOSITION: {details.get('composition', '').strip()}",
                f"HISTORICAL GUIDANCE: {details.get('historical_notes', '').strip()}",
                f"MOOD: {details.get('mood', '').strip()}",
                "FINAL CHECK: One coherent widescreen 16:9 hand-drawn stickman documentary illustration. No text, no labels, no watermark.",
            ])
            planned.append({
                **window,
                "visual_brief": brief,
                "composition": details.get("composition", ""),
                "historical_notes": details.get("historical_notes", ""),
                "mood": details.get("mood", ""),
                "image_prompt": prompt,
                "image_file": f"scene_{len(planned) + 1:04d}.png",
                "status": "planned",
            })

    if len(planned) != len(windows):
        raise RuntimeError(f"Incomplete storyboard: expected {len(windows)}, received {len(planned)}.")
    for scene in planned:
        if scene["duration"] > max_seconds + 0.001:
            raise RuntimeError(f"Scene exceeds maximum duration: {scene['id']}")
    write_json_atomic(output_path, {
        "version": 1,
        "source_transcript": str(transcript_path),
        "model": model,
        "max_scene_seconds": max_seconds,
        "scene_count": len(planned),
        "created_at_unix": time.time(),
        "scenes": planned,
    })
    print(f"Saved {len(planned)} scenes to {output_path}")
    return planned
