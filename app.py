from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from pipeline.audio_analyzer import transcribe_audio
from pipeline.config import ROOT, Settings, project_path
from pipeline.image_generator import generate_images
from pipeline.io_utils import read_json, write_json_atomic
from pipeline.quality import make_contact_sheet, write_timeline
from pipeline.scene_planner import plan_scenes

st.set_page_config(page_title="Stickman Documentary Studio", page_icon="🎬", layout="wide")
st.title("Stickman Documentary Studio")
st.caption("Turn narration into a timed storyboard and consistent hand-drawn stickman illustrations.")

with st.sidebar:
    st.header("Gemini settings")
    api_key = st.text_input("Gemini API key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    text_model = st.text_input("Storyboard model", value=os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash"))
    image_model = st.text_input("Image model", value=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image"))
    transcribe_model = st.text_input("Transcription model", value=os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-3.5-transcribe"))
    max_seconds = st.slider("Maximum image duration (seconds)", 2.0, 5.0, 4.8, 0.1)
    image_size = st.selectbox("Image resolution", ["1K", "2K", "4K"], index=0)
    st.caption("Use a paid Gemini API tier for large image batches. Image generation costs are charged by Google.")

AUDIO_DIR = ROOT / "assets" / "narration"
REF_DIR = ROOT / "assets" / "references"
TRANSCRIPT_PATH = ROOT / "output" / "transcription" / "transcript.json"
SCENES_PATH = ROOT / "output" / "storyboard" / "scenes.json"
IMAGE_DIR = ROOT / "output" / "images"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("1. Narration audio")
    audio_upload = st.file_uploader("Upload ElevenLabs MP3 or WAV", type=["mp3", "wav", "m4a", "aac"])
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
        existing_audio = sorted(p for p in AUDIO_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"})
        if existing_audio:
            audio_path = existing_audio[0]
            st.info(f"Using existing audio: {audio_path.name}")
with col_b:
    st.subheader("2. Stickman style references")
    refs_upload = st.file_uploader("Upload clean reference images (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    if refs_upload:
        for item in refs_upload:
            (REF_DIR / item.name).write_bytes(item.getbuffer())
        st.success(f"Saved {len(refs_upload)} reference image(s).")
    existing_refs = [p for p in REF_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    st.caption(f"{len(existing_refs)} reference image(s) currently available.")
    if existing_refs:
        st.image([str(p) for p in existing_refs[:4]], width=130)

st.divider()

if st.button("Step 1 + 2: Transcribe audio and create storyboard", type="primary", disabled=not (api_key and audio_path.exists())):
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GEMINI_TRANSCRIBE_MODEL"] = transcribe_model
    os.environ["GEMINI_TEXT_MODEL"] = text_model
    os.environ["GEMINI_IMAGE_MODEL"] = image_model
    try:
        with st.status("Analyzing narration...", expanded=True) as status:
            settings = Settings.from_env()
            st.write("Uploading narration and extracting timed narration segments with Gemini 3.8 Flash...")
            transcript = transcribe_audio(audio_path, TRANSCRIPT_PATH, settings.transcribe_model, settings.api_key, max_seconds)
            st.write(f"Transcribed {transcript['word_count']} words; audio duration {transcript['duration_seconds']:.1f}s.")
            st.write("Creating one historically grounded image brief for each timed scene...")
            scenes = plan_scenes(
                TRANSCRIPT_PATH, SCENES_PATH, settings.text_model, settings.api_key,
                max_seconds, settings.min_scene_seconds, settings.planner_batch_size,
            )
            status.update(label=f"Storyboard ready: {len(scenes)} scenes", state="complete")
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if SCENES_PATH.exists():
    doc = read_json(SCENES_PATH)
    scenes = doc.get("scenes", [])
    generated = sum((IMAGE_DIR / s.get("image_file", "")).exists() for s in scenes)
    st.subheader("3. Review storyboard and generate images")
    m1, m2, m3 = st.columns(3)
    m1.metric("Scenes planned", len(scenes))
    m2.metric("Images generated", generated)
    m3.metric("Max scene duration", f"{doc.get('max_scene_seconds', max_seconds):.1f}s")

    st.warning("Review scene prompts before generating the full set. Image generation uses API credits.")
    scene_ids = [scene["id"] for scene in scenes]
    selected_id = st.selectbox("Inspect a scene", scene_ids)
    selected_scene = next(s for s in scenes if s["id"] == selected_id)
    st.markdown(f"**{selected_scene['start']:.2f}s – {selected_scene['end']:.2f}s** · {selected_scene['duration']:.2f} seconds")
    st.write("**Narration**")
    st.write(selected_scene.get("narration", ""))
    st.write("**Visual brief**")
    st.write(selected_scene.get("visual_brief", ""))
    edited_prompt = st.text_area("Full image prompt (editable)", value=selected_scene.get("image_prompt", ""), height=260)
    if st.button("Save prompt edits"):
        selected_scene["image_prompt"] = edited_prompt
        write_json_atomic(SCENES_PATH, doc)
        st.success("Prompt updated.")

    c1, c2, c3 = st.columns(3)
    with c1:
        test_count = st.number_input("Test image count", min_value=1, max_value=min(10, len(scenes)), value=min(3, len(scenes)))
        if st.button("Generate first test images", disabled=not api_key):
            os.environ["GEMINI_API_KEY"] = api_key
            try:
                settings = Settings.from_env()
                generate_images(SCENES_PATH, IMAGE_DIR, REF_DIR, image_model, api_key,
                                settings.image_aspect_ratio, image_size, settings.image_retries,
                                int(test_count), False)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with c2:
        if st.button("Generate selected scene", disabled=not api_key):
            os.environ["GEMINI_API_KEY"] = api_key
            try:
                settings = Settings.from_env()
                generate_images(SCENES_PATH, IMAGE_DIR, REF_DIR, image_model, api_key,
                                settings.image_aspect_ratio, image_size, settings.image_retries,
                                None, True, {selected_id})
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with c3:
        if st.button("Generate all remaining images", disabled=not api_key):
            os.environ["GEMINI_API_KEY"] = api_key
            try:
                settings = Settings.from_env()
                generate_images(SCENES_PATH, IMAGE_DIR, REF_DIR, image_model, api_key,
                                settings.image_aspect_ratio, image_size, settings.image_retries)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("4. Image gallery")
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
        if (ROOT / "output/storyboard/contact_sheet.jpg").exists():
            st.image(str(ROOT / "output/storyboard/contact_sheet.jpg"), use_container_width=True)
else:
    st.info("Upload the narration, add a Gemini API key, then run Step 1 + 2. No images are generated until you explicitly press a generation button.")
