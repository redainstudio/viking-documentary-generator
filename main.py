from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from pipeline.audio_analyzer import transcribe_audio
from pipeline.config import ROOT, Settings, project_path
from pipeline.image_generator import generate_images
from pipeline.io_utils import read_json
from pipeline.quality import make_contact_sheet, write_timeline
from pipeline.scene_planner import plan_scenes
from pipeline.video_assembler import assemble_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn narration audio into timed stickman documentary illustrations and an ordered video."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    transcribe = sub.add_parser("transcribe", help="Transcribe narration and extract word-level timestamps.")
    transcribe.add_argument("--audio", default="assets/narration/narration.mp3")
    transcribe.add_argument("--output", default="output/transcription/transcript.json")

    plan = sub.add_parser("plan", help="Create one timed image prompt per narration window.")
    plan.add_argument("--transcript", default="output/transcription/transcript.json")
    plan.add_argument("--output", default="output/storyboard/scenes.json")

    generate = sub.add_parser("generate", help="Generate storyboard images. This command uses image API credits.")
    generate.add_argument("--scenes", default="output/storyboard/scenes.json")
    generate.add_argument("--output-dir", default="output/images")
    generate.add_argument("--references", default="assets/references")
    generate.add_argument("--limit", type=int, default=None, help="Generate only the first N scenes (use for a low-cost test).")
    generate.add_argument("--overwrite", action="store_true", help="Regenerate images even when files already exist.")
    generate.add_argument("--ids", nargs="+", default=None, help="Generate only these scene IDs, e.g. scene_0001 scene_0008.")

    contact = sub.add_parser("contact-sheet", help="Create an overview image to review scene order and quality.")
    contact.add_argument("--scenes", default="output/storyboard/scenes.json")
    contact.add_argument("--image-dir", default="output/images")
    contact.add_argument("--output", default="output/storyboard/contact_sheet.jpg")
    contact.add_argument("--columns", type=int, default=3)

    timeline = sub.add_parser("timeline", help="Export a CSV timeline for external editors.")
    timeline.add_argument("--scenes", default="output/storyboard/scenes.json")
    timeline.add_argument("--image-dir", default="output/images")
    timeline.add_argument("--output", default="output/storyboard/timeline.csv")

    assemble = sub.add_parser("assemble", help="Assemble scene stills with narration using FFmpeg.")
    assemble.add_argument("--scenes", default="output/storyboard/scenes.json")
    assemble.add_argument("--image-dir", default="output/images")
    assemble.add_argument("--audio", default="assets/narration/narration.mp3")
    assemble.add_argument("--output", default="output/video/documentary.mp4")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.command == "transcribe":
            transcribe_audio(
                project_path(args.audio), project_path(args.output),
                settings.transcribe_model, settings.api_key,
            )
        elif args.command == "plan":
            plan_scenes(
                project_path(args.transcript), project_path(args.output),
                settings.text_model, settings.api_key,
                settings.max_scene_seconds, settings.min_scene_seconds,
                settings.planner_batch_size,
            )
        elif args.command == "generate":
            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit must be a positive integer.")
            generate_images(
                project_path(args.scenes), project_path(args.output_dir), project_path(args.references),
                settings.image_model, settings.api_key, settings.image_aspect_ratio,
                settings.image_size, settings.image_retries, args.limit, args.overwrite,
                set(args.ids) if args.ids else None,
            )
        elif args.command == "contact-sheet":
            make_contact_sheet(
                project_path(args.scenes), project_path(args.image_dir),
                project_path(args.output), max(1, args.columns),
            )
        elif args.command == "timeline":
            write_timeline(project_path(args.scenes), project_path(args.output), project_path(args.image_dir))
        elif args.command == "assemble":
            assemble_video(
                project_path(args.scenes), project_path(args.image_dir),
                project_path(args.audio), project_path(args.output),
            )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
