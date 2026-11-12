# Viking Documentary Generator

A local, narration-first tool that turns documentary audio into a timed stickman storyboard, historically specific image prompts, generated 16:9 illustrations, and an ordered video timeline. Reusable for future documentary topics.

## Workflow
1. Upload an ElevenLabs MP3/WAV and optional clean stickman style references.
2. Transcribe the existing audio locally with MLX Whisper on Apple Silicon. No audio is uploaded to a transcription API.
3. Extract word-level timestamps and divide the complete timeline into continuous scenes (default 4.8 seconds; hard maximum 5 seconds).
4. Plan one historically grounded visual per timed window with Gemini 3.8 Flash.
5. Review/edit prompts, generate test images, then generate the remaining images with Gemini 3.1 Flash Image.
6. Review images, regenerate individual scenes, export a CSV timeline, and optionally assemble an MP4 with FFmpeg.

The storyboard planner preserves exact input windows: it may not add, remove, merge, split, or reorder them. Each prompt includes shared art direction plus a specific subject, action, setting, composition, historical guidance and mood.

## Requirements
- macOS with Apple Silicon (M1/M2/M3/M4) for MLX Whisper
- Python 3.11+ (Python 3.12 recommended)
- Gemini API key for storyboard planning and image generation only
- FFmpeg on PATH for MP4 assembly (`brew install ffmpeg` on macOS)

## macOS quick start
```bash
chmod +x run_mac.command
./run_mac.command
```
The launcher creates `.venv`, installs requirements and starts Streamlit at `http://localhost:8501`. On Apple Silicon, the requirements install MLX Whisper. The first transcription downloads the Whisper model weights from Hugging Face (about 1.6 GB for large-v3-turbo); subsequent runs use the cached model. Add your Gemini API key in the sidebar for storyboard planning and image generation. Local transcription does not need that key.

Alternatively:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
brew install ffmpeg
cp .env.example .env  # add GEMINI_API_KEY for storyboard/image stages
```

## CLI workflow
```bash
source .venv/bin/activate
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
- Local transcription: `mlx-community/whisper-large-v3-turbo` via MLX Whisper
- Storyboard planning: `gemini-3.8-flash`
- Image generation: `gemini-3.1-flash-image`

Whisper's first run requires an internet connection to download model weights. Transcription itself runs locally on the Mac. See the [MLX Whisper documentation](https://github.com/ml-explore/mlx-examples/tree/main/whisper) for supported options and models.

## Project outputs
- `output/transcription/transcript.json` — transcript and word-level timestamps
- `output/storyboard/scenes.json` — scene timings, narration and complete prompts
- `output/images/scene_0001.png` — numbered images in narration order
- `output/storyboard/timeline.csv` — edit-friendly timeline
- `output/storyboard/contact_sheet.jpg` — image overview for QA
- `output/video/documentary.mp4` — assembled video

## Visual direction
The default style follows the provided visual references: hand-drawn 2D documentary illustrations, white circular stickman heads, expressive minimal faces, slim black limbs, bold imperfect ink outlines, detailed historical environments and restrained cinematic colour. Characters remain stick figures when wearing period clothing or armour. No routine horned helmets, modern objects, invented text, captions, watermarks or collages. Mythology is depicted as mythology, not established history.

Add clean crops of reference artwork to `assets/references/`. Avoid screenshots containing YouTube controls or unrelated UI. Reference images are sent to Gemini during generation and count as API input.

## Privacy and cost controls
- Transcription runs locally on Apple Silicon; the audio is not sent to Gemini or ElevenLabs for transcription.
- The Whisper model is downloaded once and cached locally.
- Gemini API usage is limited to storyboard planning and image generation.
- API keys, local audio, references and generated outputs are git-ignored.
- Planning does not generate images; image generation happens only when explicitly requested.
- Use `generate --limit 3` or the UI test-image button before a full batch.
- Existing images are skipped unless regeneration is requested.

## Tests
```bash
pytest
```
