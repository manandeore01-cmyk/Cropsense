import io
import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.models import load_model

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "cnn_model.h5"
LOGO_PATH = "crop_sense.png"
PATCH_SIZE = 32

st.set_page_config(
    page_title="CropSense",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🌾",
    layout="wide"
)

# ==============================
# STYLING
# ==============================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1d1f25 0%, #111217 45%, #0b0c10 100%);
        color: #f5f5f5;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    .main-shell {
        background: rgba(20, 21, 27, 0.88);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 26px;
        overflow: hidden;
        box-shadow: 0 20px 70px rgba(0,0,0,0.45);
        backdrop-filter: blur(10px);
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 24px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .brand-title {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }

    .brand-sub {
        margin: 6px 0 0 0;
        color: #b8bbc7;
        font-size: 1rem;
    }

    .status-pill {
        padding: 10px 18px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        color: #d8fbd2;
        font-weight: 600;
        font-size: 0.98rem;
    }

    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #57d26c;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 0 12px rgba(87,210,108,0.7);
    }

    .content-pad {
        padding: 20px;
    }

    .panel {
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 18px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .panel-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .soft-text {
        color: #aeb3bf;
        font-size: 0.97rem;
    }

    .legend-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
        font-size: 1rem;
    }

    .legend-box {
        width: 14px;
        height: 14px;
        border-radius: 4px;
        display: inline-block;
    }

    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 16px;
        text-align: center;
    }

    .metric-label {
        color: #aeb3bf;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
    }

    .summary-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .summary-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .summary-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 1rem;
    }

    .summary-row:last-child {
        border-bottom: none;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .caption-bar {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 14px 18px;
        margin-top: 12px;
        font-weight: 600;
        font-size: 1.05rem;
    }

    .footer-note {
        color: #97a0ad;
        font-size: 0.95rem;
        margin-top: 10px;
    }

    .stButton>button, .stDownloadButton>button {
        width: 100%;
        border-radius: 14px;
        border: none;
        background: linear-gradient(90deg, #3f9b45, #63c75c);
        color: white;
        font-weight: 700;
        padding: 0.8rem 1rem;
        box-shadow: 0 10px 25px rgba(80, 190, 100, 0.25);
    }

    .stFileUploader {
        background: rgba(255,255,255,0.025);
        border-radius: 14px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 16px;
        border-radius: 16px;
    }

    .stAlert {
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# MODEL LOAD
# ==============================
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH, compile=False)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: cnn_model.h5")
    st.stop()

try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Error loading CNN model: {e}")
    st.stop()

# ==============================
# HELPERS
# ==============================
def get_color(pred):
    if pred == 0:
        return (0, 0, 255)      # red
    if pred == 1:
        return (0, 255, 255)    # yellow
    return (0, 255, 0)          # green

def analyze_with_cnn(img, patch_size, model):
    h, w, _ = img.shape
    output_img = img.copy()
    counts = [0, 0, 0]

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y + patch_size, x:x + patch_size]

            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patch_input = cv2.resize(patch, (64, 64))
                patch_input = patch_input.astype("float32") / 255.0
                patch_input = np.expand_dims(patch_input, axis=0)

                pred = int(np.argmax(model.predict(patch_input, verbose=0), axis=1)[0])
                counts[pred] += 1

                color = get_color(pred)
                overlay = output_img.copy()
                cv2.rectangle(overlay, (x, y), (x + patch_size, y + patch_size), color, -1)
                cv2.addWeighted(overlay, 0.28, output_img, 0.72, 0, output_img)

    return output_img, counts

def build_pie_chart(low, medium, high):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#14151b")
    ax.set_facecolor("#14151b")
    labels = ["Low Yield", "Medium Yield", "High Yield"]
    sizes = [low, medium, high]
    colors = ["#ff5a5a", "#ffca3a", "#58d26a"]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white", "fontsize": 11}
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.axis("equal")
    ax.set_title("Yield Distribution", color="white", fontsize=15, pad=16)
    return fig

def create_pdf_report(uploaded_name, output_rgb, pie_fig, counts, percentages):
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig1, ax1 = plt.subplots(figsize=(11, 8.5))
        ax1.imshow(output_rgb)
        ax1.axis("off")
        ax1.set_title(f"CropSense CNN Output\nImage: {uploaded_name}", fontsize=16)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(11, 8.5))
        ax2.axis("off")
        summary_text = (
            "CropSense CNN Analysis Report\n\n"
            f"Image Name: {uploaded_name}\n"
            f"Patch Size: {PATCH_SIZE}\n"
            f"Total Patches: {sum(counts)}\n\n"
            f"Low Yield: {counts[0]} patches ({percentages[0]:.2f}%)\n"
            f"Medium Yield: {counts[1]} patches ({percentages[1]:.2f}%)\n"
            f"High Yield: {counts[2]} patches ({percentages[2]:.2f}%)"
        )
        ax2.text(0.05, 0.95, summary_text, va="top", fontsize=14)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        pdf.savefig(pie_fig, bbox_inches="tight")

    pdf_buffer.seek(0)
    return pdf_buffer

# ==============================
# HEADER
# ==============================
logo_html = ""
if os.path.exists(LOGO_PATH):
    logo_html = f'<img src="data:image/png;base64,{__import__("base64").b64encode(open(LOGO_PATH, "rb").read()).decode()}" width="180">'
else:
    logo_html = '<div style="font-size:2rem;font-weight:800;"><span style="color:#4caf50;">Crop</span><span style="color:#f59e0b;">Sense</span></div>'

st.markdown(f"""
<div class="main-shell">
    <div class="topbar">
        <div class="brand-wrap">
            <div>{logo_html}</div>
            <div>
                <div class="brand-title">AI-Powered <span style="color:#d9d9d9;font-weight:500;">Crop Health Intelligence</span></div>
                <div class="brand-sub">CNN-based smart crop analysis</div>
            </div>
        </div>
        <div class="status-pill"><span class="status-dot"></span>Status: Ready</div>
    </div>
    <div class="content-pad">
""", unsafe_allow_html=True)

# ==============================
# LAYOUT
# ==============================
left_col, center_col, right_col = st.columns([1.15, 2.9, 1.0])

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Upload Farm Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-text">Upload an aerial farm image for CNN analysis.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    analyze = st.button("Analyze Crop Health")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    st.markdown('<div class="panel-title" style="font-size:1.2rem;">Legend</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-row"><span class="legend-box" style="background:#ff5a5a;"></span> Low Yield</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-row"><span class="legend-box" style="background:#ffca3a;"></span> Medium Yield</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-row"><span class="legend-box" style="background:#58d26a;"></span> High Yield</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


with center_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analyzing crop health...</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-text">Patch-wise CNN yield segmentation with fixed analysis scale.</div>', unsafe_allow_html=True)

    output_rgb = None
    pie_fig = None
    pdf_buffer = None
    counts = [0, 0, 0]
    low = medium = high = 0.0
    total = 0

    if uploaded_file is not None and analyze:
        temp_path = None
        try:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            img = cv2.imread(temp_path)
            if img is None:
                st.error("Unable to read image.")
            else:
                with st.spinner("Running CNN analysis..."):
                    output_img, counts = analyze_with_cnn(img, PATCH_SIZE, model)

                total = sum(counts)
                if total == 0:
                    st.warning("No valid patches found.")
                else:
                    low = counts[0] / total * 100
                    medium = counts[1] / total * 100
                    high = counts[2] / total * 100

                    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                    pie_fig = build_pie_chart(low, medium, high)
                    pdf_buffer = create_pdf_report(
                        uploaded_file.name,
                        output_rgb,
                        pie_fig,
                        counts,
                        [low, medium, high]
                    )

                    st.progress(100)
                    st.image(output_rgb, use_container_width=True)

                    st.markdown(
                        f"""
                        <div class="caption-bar">
                            <span style="color:#58d26a;">High</span> {high:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp;
                            <span style="color:#ffca3a;">Medium</span> {medium:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp;
                            <span style="color:#ff5a5a;">Low</span> {low:.1f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        st.info("Upload an image and click Analyze Crop Health.")

    st.markdown('</div>', unsafe_allow_html=True)

    if pie_fig is not None:
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.pyplot(pie_fig)
        if pdf_buffer is not None:
            st.download_button(
                "Download CNN PDF Report",
                data=pdf_buffer,
                file_name="CropSense_CNN_Report.pdf",
                mime="application/pdf"
            )
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">Result Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f'''
        <div class="summary-row">
            <span class="chip"><span class="legend-box" style="background:#58d26a;"></span> High Yield</span>
            <span>{high:.1f}%</span>
        </div>
        <div class="summary-row">
            <span class="chip"><span class="legend-box" style="background:#ffca3a;"></span> Medium Yield</span>
            <span>{medium:.1f}%</span>
        </div>
        <div class="summary-row">
            <span class="chip"><span class="legend-box" style="background:#ff5a5a;"></span> Low Yield</span>
            <span>{low:.1f}%</span>
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">Insights</div>', unsafe_allow_html=True)
    if total > 0:
        dominant = max(
            [("Low Yield", low), ("Medium Yield", medium), ("High Yield", high)],
            key=lambda x: x[1]
        )[0]
        st.markdown(f"- Dominant class: **{dominant}**")
        st.markdown(f"- Total patches analyzed: **{total}**")
        st.markdown(f"- Fixed patch size used: **{PATCH_SIZE}**")
        if low > 40:
            st.markdown("- Large low-yield area detected. Field inspection recommended.")
        elif high > 50:
            st.markdown("- Strong high-yield coverage detected across the field.")
        else:
            st.markdown("- Mixed crop health distribution detected.")
    else:
        st.markdown("- Analysis results will appear here after upload.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)
