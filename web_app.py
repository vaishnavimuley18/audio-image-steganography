import sys
import time
import io
import contextlib
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.traditional import lsb, dct, dwt
from src.analysis.auto_recommend import auto_recommend

# Page config
st.set_page_config(
    page_title="Multi-Modal Image Steganography",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        color: #7B1E3A;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #555555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #7B1E3A;
    }
</style>
""", unsafe_allow_html=True)

# Storage directories
UPLOAD_DIR = PROJECT_ROOT / "output" / "web" / "uploads"
PROCESSED_DIR = PROJECT_ROOT / "output" / "web" / "processed"
STEGO_DIR = PROJECT_ROOT / "output" / "web" / "stego"
EXTRACT_DIR = PROJECT_ROOT / "output" / "web" / "extracted"

for d in [UPLOAD_DIR, PROCESSED_DIR, STEGO_DIR, EXTRACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def preprocess_cover_image(cover_path: Path) -> Path:
    """Preprocess cover image to 4000x4000 RGB PNG canvas preserving aspect ratio."""
    processed_path = PROCESSED_DIR / f"{cover_path.stem}_processed.png"
    with Image.open(cover_path) as img:
        img = img.convert("RGB")
        target_size = 4000
        orig_w, orig_h = img.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        canvas.save(processed_path)
    return processed_path


def compute_metrics(cover_img: np.ndarray, stego_img: np.ndarray):
    """Compute MSE, PSNR, SSIM, and changed pixel percentage."""
    if cover_img.shape != stego_img.shape:
        if len(cover_img.shape) != len(stego_img.shape):
            cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2GRAY)
            stego_img = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
        elif cover_img.shape[2] != stego_img.shape[2]:
            cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2GRAY)
            stego_img = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)

    cover_float = cover_img.astype(np.float64)
    stego_float = stego_img.astype(np.float64)
    diff = cover_float - stego_float
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10((255 ** 2) / mse))

    if len(cover_img.shape) == 2:
        ssim = float(structural_similarity(cover_img, stego_img, data_range=255))
        changed_mask = cover_img != stego_img
    else:
        ssim = float(structural_similarity(cover_img, stego_img, channel_axis=2, data_range=255))
        changed_mask = np.any(cover_img != stego_img, axis=2)

    changed_pixels = int(np.sum(changed_mask))
    total_pixels = int(changed_mask.size)
    changed_pct = float((changed_pixels / total_pixels) * 100)

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_pct": changed_pct
    }


# Header
st.markdown('<p class="main-title">Multi-Modal Image Steganography</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Traditional (LSB, DCT, DWT) & AI-Assisted Steganography for Audio, Image, and Text Payloads</p>', unsafe_allow_html=True)

# Tabs
tab_hide, tab_extract, tab_about = st.tabs(["🔒 Hide Data (Embed)", "🔓 Extract Data", "📊 About & Samples"])

# -------------------------------------------------------------
# TAB 1: HIDE DATA
# -------------------------------------------------------------
with tab_hide:
    st.subheader("1. Select Cover Image & Payload")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🖼️ Cover Image")
        cover_source = st.radio("Cover Source:", ["Choose Sample Cover", "Upload Custom Cover"], horizontal=True, key="cover_src")
        cover_path = None

        if cover_source == "Choose Sample Cover":
            sample_covers = list((PROJECT_ROOT / "data" / "images" / "covers").glob("*.jpg"))
            if sample_covers:
                selected_sample = st.selectbox("Select sample cover:", sample_covers, format_func=lambda p: p.name)
                cover_path = selected_sample
                st.image(str(cover_path), caption=f"Selected Cover: {cover_path.name}", use_container_width=True)
            else:
                st.info("No sample covers found in data/images/covers.")
        else:
            uploaded_cover = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"], key="cov_upload")
            if uploaded_cover:
                saved_cover_path = UPLOAD_DIR / uploaded_cover.name
                saved_cover_path.write_bytes(uploaded_cover.getbuffer())
                cover_path = saved_cover_path
                st.image(str(cover_path), caption=f"Uploaded Cover: {cover_path.name}", use_container_width=True)

    with col2:
        st.markdown("##### 📦 Secret Payload")
        payload_source = st.radio("Payload Source:", ["Choose Sample Payload", "Upload Custom Payload"], horizontal=True, key="payload_src")
        payload_path = None

        if payload_source == "Choose Sample Payload":
            sample_payloads = list((PROJECT_ROOT / "data" / "payloads" / "variants").glob("*.*"))
            if sample_payloads:
                selected_payload = st.selectbox("Select sample payload:", sample_payloads, format_func=lambda p: f"{p.name} ({p.stat().st_size:,} bytes)")
                payload_path = selected_payload
            else:
                st.info("No sample payloads found in data/payloads/variants.")
        else:
            uploaded_payload = st.file_uploader("Upload Payload File", type=["wav", "txt", "png", "jpg", "jpeg"], key="pay_upload")
            if uploaded_payload:
                saved_payload_path = UPLOAD_DIR / uploaded_payload.name
                saved_payload_path.write_bytes(uploaded_payload.getbuffer())
                payload_path = saved_payload_path

        if payload_path:
            ext = payload_path.suffix.lower()
            size_kb = payload_path.stat().st_size / 1024
            st.success(f"**Payload Selected**: `{payload_path.name}` ({size_kb:.2f} KB)")
            if ext == ".wav":
                st.audio(str(payload_path))
            elif ext in [".png", ".jpg", ".jpeg"]:
                st.image(str(payload_path), caption="Payload Image Preview", width=250)
            elif ext == ".txt":
                preview_text = payload_path.read_text(errors="ignore")[:300]
                st.text_area("Payload Text Preview", preview_text, height=100, disabled=True)

    st.markdown("---")
    st.subheader("2. Method Selection & AI Recommendation")

    method_col1, method_col2 = st.columns([1, 1])

    with method_col1:
        method_choice = st.selectbox(
            "Select Embedding Method:",
            ["AUTO (AI Recommendation)", "LSB", "DCT", "DWT"]
        )

    with method_col2:
        if st.button("🤖 Get AI Recommendation", use_container_width=True):
            if not cover_path or not payload_path:
                st.error("Please select both a cover image and a payload file first!")
            else:
                with st.spinner("Analyzing cover features and payload..."):
                    try:
                        rec = auto_recommend(cover_path, payload_path)
                        rec_method = rec.get("recommended_method")
                        utility = rec.get("predicted_utility", 0.0)
                        if rec_method:
                            st.session_state["recommended_method"] = rec_method
                            st.session_state["utility"] = utility
                            st.success(f"🎯 **Recommended Method**: `{rec_method}` (Predicted Utility Score: `{utility:.4f}`)")
                        else:
                            st.warning(f"No feasible method found: {rec.get('reason', 'Payload exceeds capacity.')}")
                    except Exception as err:
                        st.error(f"Recommendation error: {err}")

        if "recommended_method" in st.session_state:
            st.info(f"AI Suggestion: **{st.session_state['recommended_method']}** (Score: {st.session_state.get('utility', 0):.4f})")

    st.markdown("---")

    # Embed Action
    if st.button("🚀 Embed Secret Payload into Cover", type="primary", use_container_width=True):
        if not cover_path or not payload_path:
            st.error("Please provide both a cover image and a payload!")
        else:
            with st.spinner("Embedding payload... (preprocessing canvas & running steganography engine)"):
                try:
                    # Resolve method
                    chosen_method = method_choice
                    if chosen_method.startswith("AUTO"):
                        rec = auto_recommend(cover_path, payload_path)
                        chosen_method = rec.get("recommended_method")
                        if not chosen_method:
                            st.error(f"Cannot embed: {rec.get('reason', 'Payload too large')}")
                            st.stop()
                    
                    # 1. Preprocess cover
                    processed_cover = preprocess_cover_image(cover_path)

                    # 2. Embed
                    output_stego_path = STEGO_DIR / f"{chosen_method.lower()}_stego_{payload_path.stem}.png"
                    start_t = time.time()

                    with io.StringIO() as log_buf, contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
                        if chosen_method == "LSB":
                            lsb.embed_payload(str(processed_cover), str(payload_path), str(output_stego_path))
                        elif chosen_method == "DCT":
                            dct.embed_payload(str(processed_cover), str(payload_path), str(output_stego_path))
                        elif chosen_method == "DWT":
                            dwt.embed_payload(str(processed_cover), str(payload_path), str(output_stego_path))
                        else:
                            st.error(f"Unsupported method: {chosen_method}")
                            st.stop()

                    elapsed = time.time() - start_t

                    # 3. Calculate metrics
                    cover_cv = cv2.imread(str(processed_cover), cv2.IMREAD_UNCHANGED)
                    stego_cv = cv2.imread(str(output_stego_path), cv2.IMREAD_UNCHANGED)
                    metrics = compute_metrics(cover_cv, stego_cv)

                    # Save in session state for extraction tab
                    st.session_state["last_stego_path"] = output_stego_path
                    st.session_state["last_method"] = chosen_method
                    st.session_state["last_metrics"] = metrics
                    st.session_state["last_elapsed"] = elapsed

                    st.success(f"🎉 Embedding Complete using **{chosen_method}** in **{elapsed:.2f} seconds**!")

                except Exception as ex:
                    st.error(f"Embedding failed: {ex}")

    # Display Results if available
    if "last_stego_path" in st.session_state and st.session_state["last_stego_path"].exists():
        st.subheader("3. Embedding Results & Quality Metrics")
        stego_file = st.session_state["last_stego_path"]
        metrics = st.session_state.get("last_metrics", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR (Peak Signal-to-Noise)", f"{metrics.get('psnr', 0):.2f} dB")
        m2.metric("SSIM (Structural Similarity)", f"{metrics.get('ssim', 0):.4f}")
        m3.metric("MSE (Mean Squared Error)", f"{metrics.get('mse', 0):.6f}")
        m4.metric("Changed Pixels", f"{metrics.get('changed_pct', 0):.2f}%")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.image(str(stego_file), caption=f"Generated Stego Image ({stego_file.name})", use_container_width=True)
        with res_col2:
            st.markdown("##### Download Result")
            st.download_button(
                label="📥 Download Stego Image (PNG)",
                data=stego_file.read_bytes(),
                file_name=stego_file.name,
                mime="image/png",
                use_container_width=True
            )
            st.info("Tip: You can now go to the **Extract Data** tab to retrieve your hidden payload from this image!")


# -------------------------------------------------------------
# TAB 2: EXTRACT DATA
# -------------------------------------------------------------
with tab_extract:
    st.subheader("Extract Secret Payload from Stego Image")

    ext_col1, ext_col2 = st.columns([1, 1])

    with ext_col1:
        stego_input_source = st.radio(
            "Stego Image Source:",
            ["Use Generated Stego Image from Tab 1", "Upload Custom Stego Image"],
            horizontal=True
        )

        stego_to_extract = None
        if stego_input_source == "Use Generated Stego Image from Tab 1":
            if "last_stego_path" in st.session_state and st.session_state["last_stego_path"].exists():
                stego_to_extract = st.session_state["last_stego_path"]
                st.success(f"Using: `{stego_to_extract.name}`")
            else:
                st.warning("No stego image generated yet. Embed data in Tab 1 first or upload below.")
        else:
            uploaded_stego = st.file_uploader("Upload Stego Image:", type=["png"], key="stego_upload")
            if uploaded_stego:
                saved_path = UPLOAD_DIR / uploaded_stego.name
                saved_path.write_bytes(uploaded_stego.getbuffer())
                stego_to_extract = saved_path

    with ext_col2:
        default_method_index = 0
        if "last_method" in st.session_state:
            method_list = ["LSB", "DCT", "DWT"]
            if st.session_state["last_method"] in method_list:
                default_method_index = method_list.index(st.session_state["last_method"])

        extract_method = st.selectbox("Extraction Algorithm:", ["LSB", "DCT", "DWT"], index=default_method_index)

    if st.button("🔓 Extract Secret Payload", type="primary", use_container_width=True):
        if not stego_to_extract or not stego_to_extract.exists():
            st.error("Please provide a valid Stego image to extract from!")
        else:
            with st.spinner("Extracting payload..."):
                try:
                    start_t = time.time()
                    with io.StringIO() as log_buf, contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
                        if extract_method == "LSB":
                            res = lsb.extract_payload(str(stego_to_extract), str(EXTRACT_DIR))
                        elif extract_method == "DCT":
                            res = dct.extract_payload(str(stego_to_extract), str(EXTRACT_DIR))
                        elif extract_method == "DWT":
                            res = dwt.extract_payload(str(stego_to_extract), str(EXTRACT_DIR))
                        else:
                            st.error("Unknown method")
                            st.stop()

                    elapsed = time.time() - start_t
                    out_path = Path(res["output_path"])
                    st.success(f"Payload successfully extracted in **{elapsed:.2f} seconds**!")
                    st.markdown(f"**Recovered file**: `{out_path.name}` ({out_path.stat().st_size:,} bytes)")

                    ext = out_path.suffix.lower()
                    if ext == ".wav":
                        st.markdown("##### 🎵 Extracted Audio Playback:")
                        st.audio(str(out_path))
                    elif ext in [".png", ".jpg", ".jpeg"]:
                        st.markdown("##### 🖼️ Extracted Image:")
                        st.image(str(out_path), width=350)
                    elif ext == ".txt":
                        st.markdown("##### 📄 Extracted Text Content:")
                        st.text_area("Extracted Text", out_path.read_text(errors="ignore"), height=150)

                    st.download_button(
                        label=f"📥 Download Extracted File ({out_path.name})",
                        data=out_path.read_bytes(),
                        file_name=out_path.name,
                        use_container_width=True
                    )

                except Exception as ex:
                    st.error(f"Extraction failed: {ex}")


# -------------------------------------------------------------
# TAB 3: ABOUT & SAMPLES
# -------------------------------------------------------------
with tab_about:
    st.subheader("About Audio-Image Steganography")
    st.markdown("""
    This project explores multimodal steganography techniques to conceal high-capacity payloads (including audio, text, and images) inside cover images.

    ### Supported Methods:
    - **LSB (Least Significant Bit)**: High capacity, embeds data directly into the least significant bit of RGB channels. Best for large audio payloads.
    - **DCT (Discrete Cosine Transform)**: Frequency-domain embedding into quantized DCT coefficients. High robustness against lossy compression.
    - **DWT (Discrete Wavelet Transform)**: Multi-resolution wavelet decomposition embedding. Robust against geometric transformations and noise.
    - **AI Auto-Recommender**: Uses machine learning to evaluate cover texture, gradient variance, entropy, and payload size to recommend the highest-utility method.
    """)
