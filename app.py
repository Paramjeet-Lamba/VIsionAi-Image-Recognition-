"""
Project 2 — Image Recognition (Streamlit Edition)
=====================================================================
A modern, two-screen Streamlit app:
  1. HOME PAGE   — landing/hero screen explaining the project
  2. CLASSIFIER  — the actual working tool (upload/predict/results)

HOW TO RUN
-----------
    pip install streamlit tensorflow pillow numpy pandas
    streamlit run app.py
"""

import io
import os
import time
import urllib.request

# Quiet down TensorFlow's C++ log spam (must be set before TF is imported)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ------------------------------------------------------------------
# Page config — MUST be the first Streamlit command.
# ------------------------------------------------------------------
st.set_page_config(
    page_title="VisionAI — Image Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# TensorFlow imports (after page config, so a missing-dependency
# error still renders inside a nice Streamlit page instead of a
# bare crash).
# ------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
        decode_predictions,
    )
    from tensorflow.keras.utils import img_to_array
    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except ImportError as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(e)


# ==================================================================
# GLOBAL STYLES — modern, card-based, gradient-accented UI
# ==================================================================
CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #1b1035 0%, #0b0c1a 45%, #05060d 100%);
        color: #EAEAF4;
    }

    /* ---------- Hero (home page) ---------- */
    .hero-wrap {
        text-align: center;
        padding: 4.5rem 1rem 2rem 1rem;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        background: rgba(124, 92, 255, 0.15);
        border: 1px solid rgba(124, 92, 255, 0.4);
        color: #B8A6FF;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1.4rem;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(90deg, #A78BFA 0%, #F472B6 50%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #A6A6C1;
        max-width: 640px;
        margin: 0 auto 2.3rem auto;
        line-height: 1.6;
    }

    /* ---------- Feature / stat cards ---------- */
    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.6rem 1.4rem;
        height: 100%;
        transition: all 0.25s ease;
    }
    .card:hover {
        border-color: rgba(167, 139, 250, 0.5);
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.06);
    }
    .card-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
    .card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem; color: #F1EEFF; }
    .card-body { color: #9C9BB8; font-size: 0.92rem; line-height: 1.5; }

    .pipeline-step {
        text-align: center;
        padding: 1rem 0.5rem;
    }
    .pipeline-num {
        width: 34px; height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #A78BFA, #60A5FA);
        color: #fff;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.5rem auto;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .pipeline-label { font-size: 0.85rem; color: #B8B8D0; font-weight: 600; }

    /* ---------- Working page ---------- */
    .section-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F1EEFF;
        margin-bottom: 0.2rem;
    }
    .section-sub {
        color: #8F8FB0;
        font-size: 0.95rem;
        margin-bottom: 1.3rem;
    }
    .result-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
    }
    .top-pred-label {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: capitalize;
    }
    .top-pred-conf {
        color: #9C9BB8;
        font-size: 0.95rem;
    }
    .bar-row { display: flex; align-items: center; margin: 0.55rem 0; gap: 0.7rem; }
    .bar-label { width: 150px; font-size: 0.85rem; color: #C9C9E0; text-transform: capitalize; text-align: right; }
    .bar-track { flex: 1; background: rgba(255,255,255,0.06); border-radius: 8px; height: 20px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #A78BFA, #60A5FA); }
    .bar-pct { width: 55px; font-size: 0.85rem; color: #E5E5F5; font-weight: 600; }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: none;
        padding: 0.6rem 1.6rem;
        background: linear-gradient(90deg, #A78BFA, #60A5FA);
        color: #0b0c1a;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 92, 255, 0.35);
        color: #0b0c1a;
    }

    .footer-note {
        text-align: center;
        color: #6E6E8C;
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-bottom: 0.5rem;
    }

    .dev-credit {
        text-align: center;
        padding-bottom: 1.8rem;
    }
    .dev-credit span {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.85rem;
        color: #9C9BC4;
        font-weight: 600;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .dev-credit span .dev-name {
        background: linear-gradient(90deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* ---------- Custom branded navbar ---------- */
    .navbar {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1.4rem 0.2rem 1.6rem 0.2rem;
        margin-bottom: 1.8rem;
    }
    .navbar-brand {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 2.1rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        padding: 0.55rem 1.8rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(167,139,250,0.18), rgba(96,165,250,0.18));
        border: 1px solid rgba(167,139,250,0.45);
        box-shadow: 0 0 40px rgba(124, 92, 255, 0.25);
        color: #F1EEFF;
    }
    .navbar-brand .brand-icon {
        font-size: 2rem;
        filter: drop-shadow(0 0 10px rgba(167,139,250,0.6));
    }
    .navbar-brand .brand-accent {
        background: linear-gradient(90deg, #A78BFA, #F472B6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .navbar-tag {
        font-size: 0.82rem;
        color: #9C9BC4;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-top: 0.7rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_navbar():
    """Visible branded header shown at the top of every page — this is
    what actually displays the app's name now that the built-in
    Streamlit header is hidden via CSS."""
    st.markdown(
        """
        <div class="navbar">
            <div class="navbar-brand">
                <span class="brand-icon">🧠</span>
                Vision<span class="brand-accent">AI</span>
            </div>
            <div class="navbar-tag">✨ Instant Image Recognition, Powered by Deep Learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dev_credit():
    """Small developer-credit badge shown at the bottom of every page."""
    st.markdown(
        """
        <div class="dev-credit">
            <span>👨‍💻 Developed by&nbsp;<span class="dev-name">Paramjeet Lamba</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================
# MODEL LOADING (cached — only runs once per app session)
# ==================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    model = MobileNetV2(weights="imagenet")
    # Warm-up call: Keras/TensorFlow "traces" (compiles) the computation
    # graph the first time it's ever called. If we skip this, that one-time
    # cost silently lands on the user's first click instead of here, during
    # loading, where a spinner is already showing. Every call after this
    # one reuses the compiled graph and runs fast.
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model(dummy_input, training=False)
    return model


FALLBACK_IMAGE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
)
FALLBACK_IMAGE_PATH = "sample_image.jpg"


@st.cache_data(show_spinner=False)
def get_sample_image_bytes() -> bytes:
    if not os.path.exists(FALLBACK_IMAGE_PATH):
        urllib.request.urlretrieve(FALLBACK_IMAGE_URL, FALLBACK_IMAGE_PATH)
    with open(FALLBACK_IMAGE_PATH, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def classify_image_bytes_cached(image_bytes: bytes):
    """Cached wrapper keyed on the raw image bytes: classifying the same
    photo twice returns the cached result instantly instead of re-running
    the network."""
    model = load_model()
    pil_image = Image.open(io.BytesIO(image_bytes))
    return classify_image(pil_image, model)


def classify_image(pil_image: Image.Image, model):
    """Resize, preprocess, and run MobileNetV2 prediction on a PIL image."""
    resized = pil_image.convert("RGB").resize((224, 224))
    image_array = img_to_array(resized)
    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_batch)

    start_time = time.time()
    # NOTE: model.predict(...) is convenient but, for a single image, it is
    # noticeably slower than calling the model directly — predict() builds a
    # small tf.data pipeline under the hood on every call. model(x) skips
    # that and goes straight to the compiled graph.
    predictions = model(processed_image, training=False).numpy()
    elapsed_ms = (time.time() - start_time) * 1000

    decoded = decode_predictions(predictions, top=5)[0]
    results = [
        {"label": label.replace("_", " "), "confidence": float(prob)}
        for (_, label, prob) in decoded
    ]
    return results, elapsed_ms, resized


# ==================================================================
# SESSION STATE — controls which "page" is shown
# ==================================================================
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page_name: str):
    st.session_state.page = page_name


# ==================================================================
# PAGE 1 — HOME
# ==================================================================
def render_home():
    render_navbar()
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-badge">🧠 Transfer Learning · MobileNetV2 · ImageNet-1000</div>
            <div class="hero-title">See What Your AI<br>Actually Sees</div>
            <div class="hero-subtitle">
                Upload any photo and watch a neural network — pretrained on
                over a million images — identify the object in it, live,
                right in your browser. No training required.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        if st.button("🚀  Launch Classifier", use_container_width=True):
            go_to("app")
            st.rerun()

    st.write("")
    st.write("")

    # ---- Feature cards ----
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="card-icon">⚡</div>
                <div class="card-title">Instant Predictions</div>
                <div class="card-body">Runs a full forward pass through a deep
                convolutional network in well under a second on a normal laptop CPU.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="card-icon">🎯</div>
                <div class="card-title">1,000 Object Classes</div>
                <div class="card-body">Recognizes everyday objects, animals,
                vehicles and more — everything MobileNetV2 learned from the
                ImageNet dataset.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
                <div class="card-icon">🔁</div>
                <div class="card-title">Zero Training Needed</div>
                <div class="card-body">Uses transfer learning — a model
                someone else already trained — instead of building one from
                scratch.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")
    st.markdown('<div class="section-title" style="text-align:center;">How it works</div>', unsafe_allow_html=True)
    st.write("")

    p1, p2, p3, p4, p5 = st.columns(5)
    steps = [
        ("1", "Upload photo"),
        ("2", "Resize to 224×224"),
        ("3", "Preprocess pixels"),
        ("4", "Run neural network"),
        ("5", "Show top-5 guesses"),
    ]
    for col, (num, label) in zip([p1, p2, p3, p4, p5], steps):
        with col:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="pipeline-num">{num}</div>
                    <div class="pipeline-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="footer-note">Built with Streamlit · TensorFlow · MobileNetV2</div>',
        unsafe_allow_html=True,
    )
    render_dev_credit()


# ==================================================================
# PAGE 2 — CLASSIFIER (the working app)
# ==================================================================
def render_classifier():
    render_navbar()
    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown('<div class="section-title">🖼️ Image Classifier</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Upload a photo, or try the sample image, and let the model guess what\'s in it.</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        st.write("")
        if st.button("← Home", use_container_width=True):
            go_to("home")
            st.rerun()

    if not TF_AVAILABLE:
        st.error(
            "TensorFlow isn't installed in this environment.\n\n"
            f"Import error: `{TF_IMPORT_ERROR}`\n\n"
            "Install it with: `pip install tensorflow`"
        )
        return

    with st.spinner("🔄 Loading pretrained MobileNetV2 model (first run downloads ~14 MB)..."):
        model = load_model()

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("##### 1. Choose an image")
        uploaded_file = st.file_uploader(
            "Upload a JPG or PNG photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        use_sample = st.button("🐒 Or try the sample image instead", use_container_width=True)

        image_bytes = None
        image_source_label = None

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            image_source_label = uploaded_file.name
        elif use_sample:
            with st.spinner("Downloading sample image..."):
                image_bytes = get_sample_image_bytes()
            image_source_label = "sample_image.jpg (baboon)"
            st.session_state["_last_sample_used"] = True

        # Persist last-used image across reruns (e.g. clicking sample once)
        if image_bytes is not None:
            st.session_state["current_image_bytes"] = image_bytes
            st.session_state["current_image_label"] = image_source_label

        active_bytes = st.session_state.get("current_image_bytes")
        active_label = st.session_state.get("current_image_label")

        if active_bytes is not None:
            pil_image = Image.open(io.BytesIO(active_bytes))
            st.image(pil_image, caption=active_label, use_container_width=True)
        else:
            st.info("👆 Upload a photo or click the sample-image button to begin.")

    with right_col:
        st.markdown("##### 2. Prediction results")

        if st.session_state.get("current_image_bytes") is not None:
            pil_image = Image.open(io.BytesIO(st.session_state["current_image_bytes"]))
            run = st.button("✨ Classify this image", use_container_width=True)

            if run:
                with st.spinner("Running the image through the neural network..."):
                    results, elapsed_ms, _ = classify_image_bytes_cached(
                        st.session_state["current_image_bytes"]
                    )
                st.session_state["last_results"] = results
                st.session_state["last_elapsed_ms"] = elapsed_ms

            results = st.session_state.get("last_results")
            elapsed_ms = st.session_state.get("last_elapsed_ms")

            if results:
                top = results[0]
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="top-pred-label">{top['label']}</div>
                        <div class="top-pred-conf">{top['confidence']*100:.1f}% confident · predicted in {elapsed_ms:.0f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---- Low-confidence advisory ----
                # ImageNet's 1,000 classes contain almost no people/faces
                # category — so a low top score on a portrait or selfie is
                # usually not a bug, it's the model honestly saying "none
                # of the objects I know match this well."
                if top["confidence"] < 0.5:
                    st.markdown(
                        f"""
                        <div class="card" style="margin-top:0.8rem; border-color:rgba(244,114,182,0.35);">
                            <div class="card-title">⚠️ Low confidence ({top['confidence']*100:.0f}%)</div>
                            <div class="card-body">
                                MobileNetV2 was trained on <b>ImageNet's 1,000 categories</b> —
                                mostly everyday objects, animals, and vehicles. It has
                                <b>no "person" or "face" category</b>, so photos of people
                                often get low-confidence, odd-looking guesses like clothing
                                items instead. That's expected, not an error.<br><br>
                                If you're trying to find/identify a <b>face</b>, that's a
                                different technique (face detection, e.g. OpenCV Haar
                                Cascades) — this model is built for object recognition, not faces.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ---- Broad-category clarification ----
                # A handful of ImageNet labels are commonly mistaken for
                # brand names when they're actually generic body-style
                # categories. Flag these so a correct prediction doesn't
                # look like a wrong one.
                CATEGORY_NOTES = {
                    "jeep": "ImageNet's \"jeep, landrover\" class is a generic body-style "
                            "category for boxy 4×4 SUVs — it covers Jeeps, Land Rover "
                            "Defenders, and similar vehicles alike. It is not brand-specific.",
                    "minivan": "This is a generic vehicle body-style category, not a "
                               "specific make or model.",
                    "pickup": "This is a generic vehicle body-style category, not a "
                              "specific make or model.",
                    "sports car": "This is a generic body-style category covering many "
                                  "brands, not a specific make or model.",
                    "convertible": "This is a generic body-style category covering many "
                                   "brands, not a specific make or model.",
                    "limousine": "This is a generic body-style category, not a specific "
                                 "make or model.",
                    "cab": "This is a generic \"taxi/car\" category, not a specific make "
                           "or model.",
                    "beach wagon": "This is ImageNet's label for station wagon/estate-style "
                                   "vehicles — a body-style category, not a specific brand.",
                }
                if top["label"] in CATEGORY_NOTES:
                    st.markdown(
                        f"""
                        <div class="card" style="margin-top:0.8rem; border-color:rgba(96,165,250,0.35);">
                            <div class="card-title">ℹ️ About this label</div>
                            <div class="card-body">{CATEGORY_NOTES[top['label']]}
                            ImageNet has no brand/model-level classes at all — telling
                            specific car brands or models apart needs a different,
                            specialized model.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.markdown("**Top 5 guesses**")
                for r in results:
                    pct = r["confidence"] * 100
                    st.markdown(
                        f"""
                        <div class="bar-row">
                            <div class="bar-label">{r['label']}</div>
                            <div class="bar-track"><div class="bar-fill" style="width:{pct}%;"></div></div>
                            <div class="bar-pct">{pct:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("📊 View as data table"):
                    df = pd.DataFrame(results)
                    df["confidence"] = (df["confidence"] * 100).round(2)
                    df.columns = ["Label", "Confidence (%)"]
                    st.dataframe(df, use_container_width=True, hide_index=True)

                st.caption(
                    "⚠️ A high confidence score means the pattern strongly matched "
                    "something the model learned during training — it is **not** "
                    "a guarantee of correctness."
                )
            else:
                st.info("Click **Classify this image** to run the prediction.")
        else:
            st.write("")
            st.markdown(
                '<div class="card"><div class="card-body">Results will appear here once you choose an image on the left.</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="footer-note">MobileNetV2 · pretrained on ImageNet (1,000 classes) · transfer learning demo</div>',
        unsafe_allow_html=True,
    )
    render_dev_credit()


# ==================================================================
# ROUTER
# ==================================================================
if st.session_state.page == "home":
    render_home()
else:
    render_classifier()
