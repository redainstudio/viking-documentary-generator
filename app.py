from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from pipeline.audio_analyzer import transcribe_audio
from pipeline.config import ROOT, Settings, require_gemini_api_key
from pipeline.image_generator import generate_images
from pipeline.io_utils import read_json, write_json_atomic
from pipeline.quality import make_contact_sheet, write_timeline
from pipeline.scene_planner import plan_scenes

st.set_page_config(page_title="Stickman Documentary Studio", page_icon="🎬", layout="wide")
st.title("Stickman Documentary Studio")
st.caption("Turn narration into a timed storyboard and consistent hand-drawn stickman illustrations.")

with st.sidebar:
    st.header("Local Whisper settings")
    whisper_model = st.text_input(
        "Whisper model",
        value=os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"),
        help="Runs locally on Apple Silicon. The first run downloads the model; later runs use its cache.",
    )
    whisper_language = st.text_input(
        "Narration language",
        value=os.getenv("WHISPER_LANGUAGE", "en"),
        help="Use 'en' for English narration or leave empty for automatic language detection.",
    )
    st.caption("Audio transcription runs locally on your Mac. It does not use Gemini or ElevenLabs transcription credits.")

    st.divider()
    st.header("Gemini settings")
    api_key = st.text_input("Gemini API key (storyboard + images)", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    text_model = st.text_input("Storyboard model", value=os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash"))
    image_model = st.text_input("Image model", value=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image"))
    st.caption("Gemini is used only to create historical scene descriptions and generate images.")
    max_seconds = st.slider("Maximum image duration (seconds)", 2.0, 5.0, 4.8, 0.1)
    image_size = st.selectbox("Image resolution", ["1K", "2K", "4K"], index=0)
    st.caption("Image generation costs are charged by Google.")

AUDIO_DIR = ROOT / "assets" / "narration"
REF_DIR = ROOT / "assets" / "references"
TRANSCRIPT_PATH = ROOT / "output" / "transcription" / "transcript.json"
SCENES_PATH = ROOT / "output" / "storyboard" / "scenes.json"
IMAGE_DIR = ROOT / "output" / "images"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
SCENES_PATH.parent.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("1. Narration audio")
    audio_upload = st.file_uploader("Upload existing narration MP3 or WAV", type=["mp3", "wav", "m4a", "aac"])
    audio_path = AUDIO_DIR / "narration.mp3"
    if audio_upload:
        suffix = Path(audio_upload.name).suffix.lower() or ".mp3"
        for old_audio in AUDIO_DIR.iterdir():
            if old_audio.is_file() and old_audio.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"}:
                old_audio.unlink()
        audio_path = AUDIO_DIR / f"narration{suffix}"
        audio_path.write_bytes(audio_upload.getbuffer())
        st.audio(audio_upload)
        st.success(f"Saved locally as {audio_path.name}")
    else:
        existing_audio = sorted(
            p for p in AUDIO_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"}
        )
        if existing_audio:
            audio_path = existing_audio[0]
            st.info(f"Using existing audio: {audio_path.name}")
with col_b:
    st.subheader("2. Stickman style references")
    refs_upload = st.file_uploader(
        "Upload clean reference images (PNG/JPG/WebP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if refs_upload:
        for item in refs_upload:
            (REF_DIR / item.name).write_bytes(item.getbuffer())
        st.success(f"Saved {len(refs_upload)} reference image(s).")
    existing_refs = [p for p in REF_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    st.caption(f"{len(existing_refs)} reference image(s) currently available.")
    if existing_refs:
        st.image([str(p) for p in existing_refs[:4]], width=130)

st.divider()

st.subheader("3. Transcribe the existing narration locally")
if st.button("Step 1: Transcribe audio with local Whisper", type="primary", disabled=not audio_path.exists()):
    try:
        with st.status("Running local Whisper transcription...", expanded=True) as status:
            st.write("Loading MLX Whisper on Apple Silicon. The first run may download the model weights.")
            transcript = transcribe_audio(
                audio_path,
                TRANSCRIPT_PATH,
                model=whisper_model.strip() or "mlx-community/whisper-large-v3-turbo",
                language=whisper_language.strip(),
            )
            st.write(f"Transcribed {transcript['word_count']} words from {transcript['duration_seconds']:.1f} seconds of audio.")
            st.write(f"Detected language: {transcript['language']}")
            status.update(label="Local transcription completed", state="complete")
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if TRANSCRIPT_PATH.exists():
    transcript_doc = read_json(TRANSCRIPT_PATH)
    st.success(
        f"Transcript ready: {transcript_doc.get('word_count', 0)} timestamped words · "
        f"{transcript_doc.get('duration_seconds', 0):.1f}s · "
        f"{transcript_doc.get('model', 'local Whisper')}"
    )
    with st.expander("Preview transcript"):
        st.write(transcript_doc.get("transcript", ""))

st.subheader("4. Create the timed storyboard")
st.caption("This stage uses Gemini to turn each 4.8-second narration window into a historically grounded visual brief. It does not generate images.")
if st.button(
    "Step 2: Create storyboard with Gemini",
    disabled=not (TRANSCRIPT_PATH.exists() and bool(api_key.strip())),
):
    try:
        key = require_gemini_api_key(api_key)
        os.environ["GEMINI_API_KEY"] = key
        os.environ["GEMINI_TEXT_MODEL"] = text_model
        settings = Settings.from_env()
        with st.status("Creating historical storyboard...", expanded=True) as status:
            st.write("Dividing word timestamps into chronological scenes (maximum 4.8 seconds each)...")
            scenes = plan_scenes(
                TRANSCRIPT_PATH,
                SCENES_PATH,
                text_model,
                key,
                max_seconds,
                settings.min_scene_seconds,
                settings.planner_batch_size,
            )
            status.update(label=f"Storyboard ready: {len(scenes)} scenes", state="complete")
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if SCENES_PATH.exists():
    doc = read_json(SCENES_PATH)
    scenes = doc.get("scenes", [])
    generated = sum((IMAGE_DIR / s.get("image_file", "")).exists() for s in scenes)
    st.subheader("5. Review storyboard and generate images")
    m1, m2, m3 = st.columns(3)
    m1.metric("Scenes planned", len(scenes))
    m2.metric("Images generated", generated)
    m3.metric("Max scene duration", f"{doc.get('max_scene_seconds', max_seconds):.1f}s")

    st.warning("Review scene prompts before generating the full set. Image generation uses Gemini API credits.")
    scene_ids = [scene["id"] for scene in scenes]
    selected_id = st.selectbox("Inspect a scene", scene_ids)
    selected_scene = next(s for s in scenes if s["id"] == selected_id)
    st.markdown(
        f"**{selected_scene['start']:.2f}s – {selected_scene['end']:.2f}s** · "
        f"{selected_scene['duration']:.2f} seconds"
    )
    st.write("**Narration**")
    st.write(selected_scene.get("narration", ""))
    st.write("**Visual brief**")
    st.write(selected_scene.get("visual_brief", ""))
    edited_prompt = st.text_area(
        "Full image prompt (editable)",
        value=selected_scene.get("image_prompt", ""),
        height=260,
    )
    if st.button("Save prompt edits"):
        selected_scene["image_prompt"] = edited_prompt
        write_json_atomic(SCENES_PATH, doc)
        st.success("Prompt updated.")

    c1, c2, c3 = st.columns(3)
    with c1:
        test_count = st.number_input(
            "Test image count",
            min_value=1,
            max_value=min(10, len(scenes)),
            value=min(3, len(scenes)),
        )
        if st.button("Generate first test images", disabled=not api_key.strip()):
            try:
                key = require_gemini_api_key(api_key)
                settings = Settings.from_env()
                generate_images(
                    SCENES_PATH, IMAGE_DIR, REF_DIR, image_model, key,
                    settings.image_aspect_ratio, image_size, settings.image_retries,
                    int(test_count), False,
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with c2:
        if st.button("Generate selected scene", disabled=not api_key.strip()):
            try:
                key = require_gemini_api_key(api_key)
                settings = Settings.from_env()
                generate_images(
                    SCENES_PATH, IMAGE_DIR, REF_DIR, image_model, key,
                    settings.image_aspect_ratio, image_size, settings.image_retries,
                    None, True, {selected_id},
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with c3:
        if st.button("Generate all remaining images", disabled=not api_key.strip()):
            try:
                key = require_gemini_api_key(api_key)
                settings = Settings.from_env()
                generate_images(
                    SCENES_PATH, IMAGE_DIR, REF_DIR, image_model, key,
                    settings.image_aspect_ratio, image_size, settings.image_retries,
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("6. Image gallery")
    for start in range(0, len(scenes), 3):
        cards = st.columns(3)
        for col, scene in zip(cards, scenes[start:start + 3]):
            with col:
                path = IMAGE_DIR / scene["image_file"]
                st.caption(f"{scene['id']} · {scene['start']:.1f}–{scene['end']:.1f}s")
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.info("Not generated yet")
                st.caption(scene.get("narration", "")[:180])

    if st.button("Export timeline CSV and contact sheet"):
        write_timeline(SCENES_PATH, ROOT / "output/storyboard/timeline.csv", IMAGE_DIR)
        make_contact_sheet(SCENES_PATH, IMAGE_DIR, ROOT / "output/storyboard/contact_sheet.jpg")
        st.success("Timeline and contact sheet exported to output/storyboard/")
        contact_sheet = ROOT / "output/storyboard/contact_sheet.jpg"
        if contact_sheet.exists():
            st.image(str(contact_sheet), use_container_width=True)
else:
    st.info("Start by transcribing your existing narration locally. No replacement voiceover will be created.")
