# Viking Documentary Generator

A local, narration-first tool that turns documentary audio into a timed stickman storyboard, historically specific image prompts, generated 16:9 illustrations, and an ordered video timeline. Reusable for future documentary topics.

## Workflow
1. Upload an ElevenLabs MP3/WAV and optional clean stickman style references.
2. Transcribe with Gemini 3.5 Transcribe and word-level timestamps.
3. Divide the complete timeline into continuous scenes (default 4.8 seconds; hard maximum 5 seconds).
4. Plan one concrete visual per timed window with Gemini 3.8 Flash.
5. Review/edit prompts, generate test images, then generate the remaining images with Gemini 3.1 Flash Image.
6. Review images, regenerate individual scenes, export a CSV timeline, and optionally assemble an MP4 with FFmpeg.

The storyboard planner preserves exact input windows: it may not add, remove, merge, split, or reorder them. Each prompt includes shared art direction plus a specific subject, action, setting, composition, historical guidance and mood.

## Requirements
- macOS, Windows or Linux
- Python 3.11+ (Python 3.12 recommended)
- Gemini API key with access to transcription, text and image models
- FFmpeg on PATH for MP4 assembly (`brew install ffmpeg` on macOS)

## macOS quick start
```bash
chmod +x run_mac.command
./run_mac.command
```
The launcher creates `.venv`, installs requirements and starts Streamlit at `http://localhost:8501`. Enter your Gemini API key in the password field. Alternatively, copy `.env.example` to `.env` and configure the key. Never commit `.env`.

## CLI workflow
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit GEMINI_API_KEY
cp /path/to/narration.mp3 assets/narration/narration.mp3
python main.py transcribe --audio assets/narration/narration.mp3
python main.py plan
python main.py generate --limit 3   # low-cost test first
python main.py contact-sheet
python main.py generate
python main.py timeline
python main.py assemble            # requires FFmpeg
```
Review `output/storyboard/scenes.json` before using image-generation credits.

## Models
- Transcription: `gemini-3.5-transcribe`
- Storyboard: `gemini-3.8-flash`
- Image generation: `gemini-3.1-flash-image`

Check current model availability, quotas, billing and regional access:
- https://ai.google.dev/gemini-api/docs/transcribe
- https://ai.google.dev/gemini-api/docs/image-generation
- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/pricing

## Project outputs
- `output/transcription/transcript.json` — transcript and word timings
- `output/storyboard/scenes.json` — scene timings, narration and complete prompts
- `output/images/scene_0001.png` — numbered images in narration order
- `output/storyboard/timeline.csv` — edit-friendly timeline
- `output/storyboard/contact_sheet.jpg` — image overview for QA
- `output/video/documentary.mp4` — assembled video

## Visual direction
The default style follows the provided visual references: hand-drawn 2D documentary illustrations, white circular stickman heads, expressive minimal faces, slim black limbs, bold imperfect ink outlines, detailed historical environments and restrained cinematic colour. Characters remain stick figures when wearing period clothing or armour. No routine horned helmets, modern objects, invented text, captions, watermarks or collages. Mythology is depicted as mythology, not established history.

Add clean crops of reference artwork to `assets/references/`. Avoid screenshots containing YouTube controls or unrelated UI. Reference images are sent to Gemini during generation and count as API input.

## Privacy and cost controls
- API keys, local audio, references and generated outputs are git-ignored.
- Planning does not generate images; image generation happens only when explicitly requested.
- Use `generate --limit 3` or the UI test-image button before a full batch.
- Existing images are skipped unless regeneration is requested.
- Audio and references are sent to Google's API. Review Google's terms and data controls.
- Model availability, quotas and pricing may change.

## Tests
```bash
pytest
```
