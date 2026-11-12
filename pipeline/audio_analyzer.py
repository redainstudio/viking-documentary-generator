from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile

from .io_utils import write_json_atomic


VIKING_VOCABULARY_PROMPT = (
    "English historical documentary narration about the Viking Age. "
    "Names and terms may include Vikings, Norse, Scandinavia, Scandinavians, "
    "Norsemen, longships, Lindisfarne, Ragnar, Rollo, Leif Erikson, Erik the Red, "
    "Harald Bluetooth, Harald Hardrada, Cnut, Jorvik, Danelaw, Valhalla, Odin, "
    "Thor, Freyja, Yggdrasil, berserkers, runes, skalds, and Vinland. "
    "Transcribe the spoken audio exactly; do not add words."
)


def _audio_duration(audio_path: Path) -> float:
    media = MutagenFile(str(audio_path))
    if not media or not getattr(media, "info", None):
        raise RuntimeError(
            "Could not read the audio duration. Check that the narration file is a valid MP3, WAV, M4A or AAC."
        )
    duration = float(media.info.length)
    if duration <= 0:
        raise RuntimeError("The narration audio has an invalid duration.")
    return duration


def _normalize_whisper_result(
    result: dict[str, Any],
    audio_duration: float,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert MLX Whisper output into the stable transcript schema used by the planner."""
    transcript = str(result.get("text") or "").strip()
    raw_segments = result.get("segments") or []
    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        try:
            segment_start = max(0.0, float(raw_segment.get("start", 0.0)))
            segment_end = min(audio_duration, float(raw_segment.get("end", audio_duration)))
        except (TypeError, ValueError):
            continue

        segment_text = str(raw_segment.get("text") or "").strip()
        if segment_text and segment_end > segment_start:
            segments.append({
                "start": round(segment_start, 3),
                "end": round(segment_end, 3),
                "text": segment_text,
            })

        for raw_word in raw_segment.get("words") or []:
            if not isinstance(raw_word, dict):
                continue
            text = str(raw_word.get("word") or raw_word.get("text") or "").strip()
            try:
                start = max(0.0, float(raw_word["start"]))
                end = min(audio_duration, float(raw_word["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            if text and end > start:
                words.append({
                    "text": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                })

    words.sort(key=lambda item: (item["start"], item["end"]))
    segments.sort(key=lambda item: (item["start"], item["end"]))

    if not transcript and words:
        transcript = " ".join(word["text"] for word in words)
    if not transcript:
        raise RuntimeError("Whisper returned no transcript text. Check the audio file and try again.")
    if not words:
        raise RuntimeError(
            "Whisper returned transcript text but no word-level timestamps. "
            "Make sure word_timestamps is enabled and the MLX Whisper package is up to date."
        )

    previous_end = 0.0
    for index, word in enumerate(words):
        if word["start"] < 0 or word["end"] < word["start"] or word["end"] > audio_duration + 0.1:
            raise RuntimeError(f"Whisper returned an invalid timestamp at word {index + 1}: {word}")
        if word["start"] + 0.05 < previous_end:
            # Keep Whisper's original timing, but make the anomaly visible in logs.
            print(f"Warning: overlapping Whisper timestamps near {word['text']!r}.")
        previous_end = max(previous_end, word["end"])

    return transcript, words, segments


def transcribe_audio(
    audio_path: Path,
    output_path: Path,
    model: str = "mlx-community/whisper-large-v3-turbo",
    language: str = "en",
) -> dict[str, Any]:
    """Transcribe narration locally with MLX Whisper and word-level timestamps.

    Runs on Apple Silicon using MLX. No audio is sent to Gemini, ElevenLabs or
    another transcription API. The model is downloaded from Hugging Face on
    first use and cached locally for later runs.
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise ValueError("Audio file is empty.")

    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "MLX Whisper is not installed in this environment. On your Apple Silicon Mac, "
            "activate the project's .venv and run: python -m pip install mlx-whisper"
        ) from exc

    duration = _audio_duration(audio_path)
    print(f"Transcribing {audio_path.name} locally with MLX Whisper...")
    print(f"Model: {model}")
    print("The first run may download the model weights. Later runs use the local cache.")

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        word_timestamps=True,
        verbose=False,
        language=language or None,
        task="transcribe",
        condition_on_previous_text=True,
        initial_prompt=VIKING_VOCABULARY_PROMPT,
        hallucination_silence_threshold=2.0,
    )

    transcript, words, segments = _normalize_whisper_result(result, duration)
    payload = {
        "source_audio": audio_path.name,
        "provider": "local",
        "model": model,
        "language": result.get("language") or language or "unknown",
        "transcript": transcript,
        "words": words,
        "segments": segments,
        "duration_seconds": duration,
        "last_word_end_seconds": max(word["end"] for word in words),
        "word_count": len(words),
        "segment_count": len(segments),
        "timestamp_note": (
            "Word-level timestamps generated locally by MLX Whisper. "
            "Review the transcript and scene boundaries before image generation."
        ),
        "created_at_unix": time.time(),
    }
    write_json_atomic(output_path, payload)

    print(f"Saved transcript and {len(words)} word timestamps to {output_path}")
    print(f"Detected language: {payload['language']}")
    print(f"Audio duration: {duration:.2f}s")
    print(f"Transcribed words: {len(words)}")
    return payload
