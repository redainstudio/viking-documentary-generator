from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    whisper_language: str = "en"
    text_model: str = "gemini-3.8-flash"
    image_model: str = "gemini-3.1-flash-image"
    max_scene_seconds: float = 4.8
    min_scene_seconds: float = 2.0
    planner_batch_size: int = 18
    image_aspect_ratio: str = "16:9"
    image_size: str = "1K"
    image_retries: int = 3
    image_concurrency: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        max_scene = float(os.getenv("MAX_SCENE_SECONDS", "4.8"))
        if not 0 < max_scene <= 5.0:
            raise ValueError("MAX_SCENE_SECONDS must be greater than 0 and no more than 5.0.")
        min_scene = float(os.getenv("MIN_SCENE_SECONDS", "2.0"))
        if not 0 < min_scene <= max_scene:
            raise ValueError("MIN_SCENE_SECONDS must be positive and no greater than MAX_SCENE_SECONDS.")
        return cls(
            api_key=key,
            whisper_model=os.getenv(
                "WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
            ).strip(),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "en").strip(),
            text_model=os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash").strip(),
            image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image").strip(),
            max_scene_seconds=max_scene,
            min_scene_seconds=min_scene,
            planner_batch_size=max(1, int(os.getenv("PLANNER_BATCH_SIZE", "18"))),
            image_aspect_ratio=os.getenv("IMAGE_ASPECT_RATIO", "16:9").strip(),
            image_size=os.getenv("IMAGE_SIZE", "1K").strip(),
            image_retries=max(1, int(os.getenv("IMAGE_RETRIES", "3"))),
            image_concurrency=max(1, int(os.getenv("IMAGE_CONCURRENCY", "1"))),
        )


def require_gemini_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key or key == "replace_me":
        raise RuntimeError(
            "GEMINI_API_KEY is required for storyboard planning and image generation, "
            "but it is not required for local Whisper transcription. Add your key in the app sidebar or .env."
        )
    return key


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path
