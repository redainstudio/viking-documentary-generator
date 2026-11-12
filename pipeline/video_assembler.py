from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .io_utils import read_json


def assemble_video(scenes_path: Path, image_dir: Path, audio_path: Path, output_path: Path, fps: int = 30) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Narration audio not found: {audio_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found. Install it with: brew install ffmpeg")
    doc = read_json(scenes_path)
    scenes = doc.get("scenes", [])
    if not scenes:
        raise ValueError("No scenes found in storyboard.")
    missing = [scene.get("image_file") for scene in scenes if not (image_dir / scene.get("image_file", "")).exists()]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Cannot assemble: {len(missing)} scene images are missing ({preview}). Generate all images first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stickman_video_") as temp_dir:
        concat_file = Path(temp_dir) / "timeline.ffconcat"
        lines = ["ffconcat version 1.0"]
        for scene in scenes:
            image = (image_dir / scene["image_file"]).resolve()
            duration = max(0.05, float(scene["end"]) - float(scene["start"]))
            lines.extend([f"file '{image.as_posix()}'", f"duration {duration:.6f}"])
        lines.append(f"file '{(image_dir / scenes[-1]['image_file']).resolve().as_posix()}'")
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        command = [
            "ffmpeg", "-hide_banner", "-y", "-safe", "0", "-f", "concat", "-i", str(concat_file),
            "-i", str(audio_path), "-vf",
            f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(output_path),
        ]
        print("Running FFmpeg video assembly...")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-4000:]}")
    print(f"Video saved: {output_path}")
