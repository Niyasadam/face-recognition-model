"""
Face Recognition Portal (FaceNet-based)
Two interfaces only: live webcam recognition and photo upload recognition.

Data collection      -> collect.py
Embedding gallery build -> train.py
"""

from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

from face_engine import detect_faces, embed_face

# ------------------------------------------------------------------
# 1. UI INITIALIZATION & THEME
# ------------------------------------------------------------------
st.set_page_config(page_title="Face Recognition Portal", layout="wide")

st.markdown("""<style>
    .stApp  { background:#070b12; color:#e0e0e0; }
    .main-title { text-align:center; font-size:2.2rem; font-weight:700;
        background:linear-gradient(135deg,#00ff96,#0575e6);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .glass { background:rgba(255,255,255,0.04); backdrop-filter:blur(12px);
        border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:1.5rem; margin-bottom:1.25rem; }
    .badge { display:inline-flex; align-items:center; gap:6px; padding:0.3rem 1rem;
        border-radius:20px; font-weight:600; font-size:0.85rem;
        background:rgba(0,255,150,0.15); color:#00ff96; border:1px solid rgba(0,255,150,0.3); }
    .badge-off { background:rgba(255,50,50,0.15); color:#ff5050; border:1px solid rgba(255,50,50,0.3); }
    .stButton>button { background:linear-gradient(135deg,#00ff96,#0575e6); color:#fff; font-weight:600; border-radius:8px; border:none; }
</style>""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Face Recognition System</h1>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. GALLERY & CONFIG
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
EMBEDDINGS_PATH = BASE_DIR / "face_embeddings.npz"

FACE_COLOR = (0, 255, 150)

# After running train.py, update this with the "Recommended
# FACE_DISTANCE_THRESHOLD" value it prints for your dataset.
FACE_DISTANCE_THRESHOLD = 0.90

# On CPU, running MTCNN + FaceNet on every webcam frame is too slow for
# smooth video. Recognition only re-runs every N frames; the most recent
# results are reused (and drawn) on the frames in between.
PROCESS_EVERY_N_FRAMES = 5

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})


@st.cache_resource(show_spinner=False)
def load_gallery():
    if not EMBEDDINGS_PATH.exists():
        return None, None, None
    try:
        data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]
        mapping = dict(data["mapping"])
        return embeddings, labels, mapping
    except Exception:
        return None, None, None


gallery_embeddings, gallery_labels, labels_map = load_gallery()
MODEL_READY = gallery_embeddings is not None and len(gallery_embeddings) > 0


def recognize(pil_img, box):
    """Embed the face at `box` and find its nearest gallery match."""
    emb = embed_face(pil_img, box)
    dists = np.linalg.norm(gallery_embeddings - emb, axis=1)
    min_idx = int(np.argmin(dists))
    min_dist = float(dists[min_idx])
    if min_dist <= FACE_DISTANCE_THRESHOLD:
        name = labels_map.get(int(gallery_labels[min_idx]), "Unknown")
    else:
        name = "Unknown"
    return name, min_dist


def draw_label(img, box, name, dist):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), FACE_COLOR, 2)
    text = f"{name} ({dist:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    ty = max(y1, th + 10)
    cv2.rectangle(img, (x1, ty - th - 10), (x1 + tw + 6, ty), FACE_COLOR, -1)
    cv2.putText(img, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 1, cv2.LINE_AA)


# ------------------------------------------------------------------
# 3. SIDEBAR STATUS
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### System Dashboard")
    if labels_map:
        subject_names = sorted(set(labels_map.values()))
        st.metric("Enrolled Profiles", len(subject_names))
        with st.expander("Enrolled subjects"):
            for name in subject_names:
                st.write(f"- {name}")
    if MODEL_READY:
        st.markdown("<span class='badge'>● Model Loaded</span>", unsafe_allow_html=True)
        st.caption(f"Gallery size: {len(gallery_embeddings)} embeddings")
    else:
        st.markdown("<span class='badge badge-off'>○ Model Offline</span>", unsafe_allow_html=True)
        st.caption("No embedding gallery found. Run `python collect.py` then `python train.py`.")

# ------------------------------------------------------------------
# 4. INTERFACE
# ------------------------------------------------------------------
tab1, tab2 = st.tabs(["📷 Live Webcam Recognition", "🖼️ Photo Recognition"])

# --- TAB 1: LIVE WEBCAM ---
with tab1:
    if not MODEL_READY:
        st.info("No embedding gallery found. Use collect.py + train.py to build one first.")
    else:
        st.markdown("<span class='badge'>● Live Tracking Feed</span>", unsafe_allow_html=True)
        st.write("")

        # Mutable state shared across frame callbacks for this stream
        stream_state = {"counter": 0, "last_results": []}

        def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            stream_state["counter"] += 1

            if stream_state["counter"] % PROCESS_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                results = []
                for box, prob in detect_faces(pil_img):
                    try:
                        name, dist = recognize(pil_img, box)
                    except Exception:
                        name, dist = "Unknown", 99.0
                    results.append((box, name, dist))
                stream_state["last_results"] = results

            for box, name, dist in stream_state["last_results"]:
                draw_label(img, box, name, dist)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="live-recognition",
            video_frame_callback=video_callback,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            async_processing=True,
        )

# --- TAB 2: PHOTO UPLOAD ---
with tab2:
    if not MODEL_READY:
        st.info("No embedding gallery found. Use collect.py + train.py to build one first.")
    else:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Upload a Photo to Identify")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        )

        if uploaded is not None:
            try:
                pil_img = Image.open(uploaded).convert("RGB")
            except Exception:
                pil_img = None
                st.error("Could not read this image file.")

            if pil_img is not None:
                faces = detect_faces(pil_img)

                if len(faces) == 0:
                    st.warning("No face detected in this image.")
                    st.image(pil_img, caption="Uploaded image", use_container_width=True)
                else:
                    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    results = []
                    for box, prob in faces:
                        name, dist = recognize(pil_img, box)
                        draw_label(img_bgr, box, name, dist)
                        results.append((name, dist))

                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

                    st.markdown("#### Detected")
                    for name, dist in results:
                        st.write(f"- **{name}** — distance {dist:.2f} "
                                 f"(lower = more confident match)")
        st.markdown("</div>", unsafe_allow_html=True)