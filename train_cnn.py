import base64
import io
import os
import tempfile
from collections import deque
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.models import load_model

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "cnn_model.h5"
LOGO_PATH = "crop_sense.png"
PATCH_SIZE = 32
MODEL_INPUT_SIZE = (64, 64)
MAX_DISPLAY_WIDTH = 1200

LOW_CLASS = 0
MEDIUM_CLASS = 1
HIGH_CLASS = 2

CLASS_LABELS = {
    LOW_CLASS: "Low Yield",
    MEDIUM_CLASS: "Medium Yield",
    HIGH_CLASS: "High Yield",
}

CLASS_COLORS_HEX = {
    LOW_CLASS: "#ff5a5a",
    MEDIUM_CLASS: "#ffca3a",
    HIGH_CLASS: "#58d26a",
}

CLASS_COLORS_BGR = {
    LOW_CLASS: (0, 0, 255),
    MEDIUM_CLASS: (0, 255, 255),
    HIGH_CLASS: (0, 255, 0),
}

st.set_page_config(
    page_title="CropSense",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🌾",
    layout="wide",
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
        max-width: 1350px;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .main-shell {
        background: rgba(20, 21, 27, 0.9);
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
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
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
        margin-bottom: 0.25rem;
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
    .summary-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .summary-title {
        font-size: 1.15rem;
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
    .compare-label {
        font-size: 1rem;
        font-weight: 700;
        color: #d5d8df;
        margin-bottom: 8px;
    }
    .subtle-divider {
        margin: 14px 0;
        border-top: 1px solid rgba(255,255,255,0.06);
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 10px 14px;
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
def get_logo_html():
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as file:
            encoded = base64.b64encode(file.read()).decode()
        return f'<img src="data:image/png;base64,{encoded}" width="180">'
    return '<div style="font-size:2rem;font-weight:800;"><span style="color:#4caf50;">Crop</span><span style="color:#f59e0b;">Sense</span></div>'

def resize_for_display(img, max_width=MAX_DISPLAY_WIDTH):
    height, width = img.shape[:2]
    if width <= max_width:
        return img
    scale = max_width / width
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def analyze_with_cnn(img, patch_size, model):
    height, width, _ = img.shape
    output_img = img.copy()
    counts = [0, 0, 0]

    patches = []
    patch_meta = []
    row_lengths = []

    for y in range(0, height, patch_size):
        row_cells = 0
        for x in range(0, width, patch_size):
            patch = img[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patch_input = cv2.resize(patch, MODEL_INPUT_SIZE)
                patch_input = patch_input.astype("float32") / 255.0
                patches.append(patch_input)
                patch_meta.append((x, y))
                row_cells += 1
        if row_cells > 0:
            row_lengths.append(row_cells)

    if not patches:
        return output_img, counts, np.array([])

    patch_batch = np.array(patches, dtype=np.float32)
    predictions = model.predict(patch_batch, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    overlay = output_img.copy()
    class_grid = []
    index = 0

    for row_length in row_lengths:
        row_preds = []
        for _ in range(row_length):
            pred = int(pred_classes[index])
            x, y = patch_meta[index]
            counts[pred] += 1
            row_preds.append(pred)
            cv2.rectangle(
                overlay,
                (x, y),
                (x + patch_size, y + patch_size),
                CLASS_COLORS_BGR[pred],
                -1,
            )
            index += 1
        class_grid.append(row_preds)

    cv2.addWeighted(overlay, 0.28, output_img, 0.72, 0, output_img)
    return output_img, counts, np.array(class_grid, dtype=object)

def find_connected_low_regions(class_grid):
    if class_grid.size == 0:
        return []

    rows, cols = class_grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    regions = []

    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or class_grid[row, col] != LOW_CLASS:
                continue

            queue = deque([(row, col)])
            visited[row, col] = True
            region_cells = []

            while queue:
                current_row, current_col = queue.popleft()
                region_cells.append((current_row, current_col))

                for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_row = current_row + d_row
                    next_col = current_col + d_col
                    if 0 <= next_row < rows and 0 <= next_col < cols:
                        if not visited[next_row, next_col] and class_grid[next_row, next_col] == LOW_CLASS:
                            visited[next_row, next_col] = True
                            queue.append((next_row, next_col))

            regions.append(region_cells)

    return regions

def outline_low_regions(output_img, low_regions, patch_size):
    highlighted = output_img.copy()
    region_boxes = []

    for region_index, region in enumerate(low_regions, start=1):
        if len(region) < 2:
            continue

        rows = [cell[0] for cell in region]
        cols = [cell[1] for cell in region]

        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        x1 = min_col * patch_size
        y1 = min_row * patch_size
        x2 = (max_col + 1) * patch_size
        y2 = (max_row + 1) * patch_size

        cv2.rectangle(highlighted, (x1, y1), (x2, y2), (80, 220, 255), 3)
        cv2.putText(
            highlighted,
            f"Zone {region_index}",
            (x1 + 6, max(24, y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        region_boxes.append(
            {
                "label": f"Zone {region_index}",
                "cells": len(region),
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            }
        )

    return highlighted, region_boxes

def detect_location(region_box, img_shape):
    height, width = img_shape[:2]
    center_x, center_y = region_box["center"]

    horizontal = "left" if center_x < width / 3 else "center" if center_x < (2 * width) / 3 else "right"
    vertical = "top" if center_y < height / 3 else "middle" if center_y < (2 * height) / 3 else "bottom"

    if vertical == "middle" and horizontal == "center":
        return "central area"
    if vertical == "middle":
        return f"{horizontal}-center area"
    if horizontal == "center":
        return f"{vertical}-center area"
    return f"{vertical}-{horizontal} area"

def generate_recommendations(percentages, region_boxes, img_shape):
    low, medium, high = percentages
    recommendations = []

    dominant_label = max(
        [("Low Yield", low), ("Medium Yield", medium), ("High Yield", high)],
        key=lambda item: item[1],
    )[0]
    recommendations.append(f"Dominant class detected: {dominant_label}.")

    if high >= 55:
        recommendations.append("Large healthy coverage is visible across the field.")
    if medium >= 35:
        recommendations.append("Moderate-yield coverage is significant, indicating mixed crop conditions.")
    if low >= 30:
        recommendations.append("Low-yield share is high. Check irrigation, nutrient supply, and pest stress.")
    elif low >= 15:
        recommendations.append("Localized low-yield pockets are present. Field inspection is recommended.")

    if region_boxes:
        largest_region = max(region_boxes, key=lambda box: box["cells"])
        location = detect_location(largest_region, img_shape)
        recommendations.append(
            f"Highest low-yield concentration detected in the {location} ({largest_region['label']})."
        )
        recommendations.append("Highlighted zones are connected groups of low-yield patches predicted by the CNN.")
    else:
        recommendations.append("No major connected low-yield stress zone was detected.")

    return recommendations

def build_pie_chart(low, medium, high):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="#14151b")
    ax.set_facecolor("#14151b")

    _, texts, autotexts = ax.pie(
        [low, medium, high],
        labels=["Low Yield", "Medium Yield", "High Yield"],
        colors=[CLASS_COLORS_HEX[LOW_CLASS], CLASS_COLORS_HEX[MEDIUM_CLASS], CLASS_COLORS_HEX[HIGH_CLASS]],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white", "fontsize": 11},
    )

    for text in texts:
        text.set_color("white")
    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")

    ax.axis("equal")
    ax.set_title("Yield Distribution", color="white", fontsize=15, pad=16)
    return fig

def build_summary_table(counts, percentages):
    return pd.DataFrame(
        {
            "Class": ["Low Yield", "Medium Yield", "High Yield"],
            "Patch Count": counts,
            "Percentage": [f"{value:.2f}%" for value in percentages],
        }
    )

def make_text_bar(label, value, total_blocks=10):
    filled = round((value / 100) * total_blocks)
    empty = total_blocks - filled
    bar = "█" * filled + "░" * empty
    return f"{label:<8} {bar} {value:.0f}%"

def create_pdf_report(image_name, original_rgb, result_rgb, pie_fig, counts, percentages, recommendations, region_boxes):
    pdf_buffer = io.BytesIO()
    generated_on = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    summary_table = build_summary_table(counts, percentages)

    zone_basis_text = (
        "Zone marking basis:\n"
        "- The original image is split into 32 x 32 pixel patches.\n"
        "- Each patch is resized to 64 x 64 pixels before CNN prediction.\n"
        "- The CNN classifies each patch as Low Yield, Medium Yield, or High Yield.\n"
        "- Connected neighboring patches predicted as Low Yield are grouped and outlined as stress zones."
    )

    bars_text = "\n".join([
        make_text_bar("High", percentages[2]),
        make_text_bar("Medium", percentages[1]),
        make_text_bar("Low", percentages[0]),
    ])

    with PdfPages(pdf_buffer) as pdf:
        fig1, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(original_rgb)
        axes[0].axis("off")
        axes[0].set_title("Original Image", fontsize=16)

        axes[1].imshow(result_rgb)
        axes[1].axis("off")
        axes[1].set_title("CNN Output with Highlighted Regions", fontsize=16)

        fig1.suptitle("CropSense Visual Comparison", fontsize=18, fontweight="bold")
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(11, 8.5))
        ax2.axis("off")
        ax2.text(0.05, 0.96, "CropSense CNN Analysis Report", fontsize=20, fontweight="bold", va="top")
        ax2.text(0.05, 0.91, f"Image Name: {image_name}", fontsize=12)
        ax2.text(0.05, 0.88, f"Generated On: {generated_on}", fontsize=12)
        ax2.text(0.05, 0.85, f"Patch extraction size: {PATCH_SIZE} x {PATCH_SIZE} pixels", fontsize=12)
        ax2.text(
            0.05,
            0.82,
            f"CNN input size after resizing: {MODEL_INPUT_SIZE[0]} x {MODEL_INPUT_SIZE[1]} pixels",
            fontsize=12,
        )
        ax2.text(0.05, 0.79, f"Total Patches Analyzed: {sum(counts)}", fontsize=12)

        ax2.text(0.05, 0.73, "Summary Table", fontsize=14, fontweight="bold")
        ax2.text(0.05, 0.70, summary_table.to_string(index=False), family="monospace", fontsize=11, va="top")

        ax2.text(0.05, 0.49, "Yield Intensity Bars", fontsize=14, fontweight="bold")
        ax2.text(0.05, 0.46, bars_text, family="monospace", fontsize=12, va="top")

        ax2.text(0.05, 0.31, "Zone Marking Basis", fontsize=14, fontweight="bold")
        ax2.text(0.05, 0.28, zone_basis_text, fontsize=11, va="top")
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(11, 8.5))
        ax3.axis("off")
        recommendation_text = "\n".join([f"- {item}" for item in recommendations])
        ax3.text(0.05, 0.95, "Recommendations", fontsize=18, fontweight="bold", va="top")
        ax3.text(0.05, 0.88, recommendation_text, fontsize=12, va="top")

        if region_boxes:
            region_text = "\n".join(
                [f"- {box['label']}: {box['cells']} connected low-yield patches" for box in region_boxes]
            )
        else:
            region_text = "- No major connected low-yield zone detected."

        ax3.text(0.05, 0.55, "Detected Stress Zones", fontsize=16, fontweight="bold")
        ax3.text(0.05, 0.49, region_text, fontsize=12, va="top")
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        pdf.savefig(pie_fig, bbox_inches="tight")

    pdf_buffer.seek(0)
    return pdf_buffer

def process_single_image(uploaded_file, model):
    temp_path = None
    try:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
        file_bytes = uploaded_file.getvalue()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError("Unable to read image.")

        display_img = resize_for_display(img)
        output_img, counts, class_grid = analyze_with_cnn(display_img, PATCH_SIZE, model)

        total = sum(counts)
        if total == 0:
            raise ValueError("No valid patches found.")

        percentages = [
            counts[LOW_CLASS] / total * 100,
            counts[MEDIUM_CLASS] / total * 100,
            counts[HIGH_CLASS] / total * 100,
        ]

        low_regions = find_connected_low_regions(class_grid)
        highlighted_output, region_boxes = outline_low_regions(output_img, low_regions, PATCH_SIZE)

        original_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(highlighted_output, cv2.COLOR_BGR2RGB)
        pie_fig = build_pie_chart(*percentages)
        recommendations = generate_recommendations(percentages, region_boxes, display_img.shape)
        pdf_buffer = create_pdf_report(
            uploaded_file.name,
            original_rgb,
            result_rgb,
            pie_fig,
            counts,
            percentages,
            recommendations,
            region_boxes,
        )

        return {
            "name": uploaded_file.name,
            "original_rgb": original_rgb,
            "result_rgb": result_rgb,
            "counts": counts,
            "percentages": percentages,
            "pie_fig": pie_fig,
            "recommendations": recommendations,
            "region_boxes": region_boxes,
            "pdf_buffer": pdf_buffer,
            "summary_table": build_summary_table(counts, percentages),
            "total": total,
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ==============================
# HEADER
# ==============================
st.markdown(
    f"""
<div class="main-shell">
    <div class="topbar">
        <div class="brand-wrap">
            <div>{get_logo_html()}</div>
            <div>
                <div class="brand-title">AI-Powered <span style="color:#d9d9d9;font-weight:500;">Crop Health Intelligence</span></div>
                <div class="brand-sub">CNN-based smart crop analysis with reporting and stress-zone highlighting</div>
            </div>
        </div>
        <div class="status-pill"><span class="status-dot"></span>Status: Ready</div>
    </div>
    <div class="content-pad">
""",
    unsafe_allow_html=True,
)

# ==============================
# LAYOUT
# ==============================
left_col, center_col, right_col = st.columns([1.15, 3.1, 1.05])

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Upload Farm Images</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-text">Upload one or more aerial farm images for CNN analysis.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Farm Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    analyze = st.button("Analyze Crop Health")

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title" style="font-size:1.2rem;">Legend</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="legend-row"><span class="legend-box" style="background:{CLASS_COLORS_HEX[LOW_CLASS]};"></span> Low Yield</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="legend-row"><span class="legend-box" style="background:{CLASS_COLORS_HEX[MEDIUM_CLASS]};"></span> Medium Yield</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="legend-row"><span class="legend-box" style="background:{CLASS_COLORS_HEX[HIGH_CLASS]};"></span> High Yield</div>', unsafe_allow_html=True)

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title" style="font-size:1.2rem;">Analysis Basis</div>', unsafe_allow_html=True)
    st.write(f"- Patch extraction size: {PATCH_SIZE} x {PATCH_SIZE} pixels")
    st.write(f"- CNN input size after resizing: {MODEL_INPUT_SIZE[0]} x {MODEL_INPUT_SIZE[1]} pixels")
    st.write("- Each patch is classified by the CNN as Low, Medium, or High Yield")
    st.write("- Connected neighboring low-yield patches are outlined as stress zones")
    st.markdown('</div>', unsafe_allow_html=True)

results = []

if uploaded_files and analyze:
    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        status.info(f"Analyzing {uploaded_file.name} ({index}/{len(uploaded_files)})")
        try:
            results.append(process_single_image(uploaded_file, model))
        except Exception as error:
            st.error(f"{uploaded_file.name}: {error}")
        progress.progress(int(index / len(uploaded_files) * 100))

    status.success("Analysis complete.")

with center_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Crop Health Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-text">Before/after comparison, PDF reporting, stress-zone highlighting, and intensity bars.</div>', unsafe_allow_html=True)

    if not uploaded_files:
        st.info("Upload one or more images and click Analyze Crop Health.")
    elif analyze and results:
        tabs = st.tabs([result["name"] for result in results])

        for tab, result in zip(tabs, results):
            with tab:
                before_col, after_col = st.columns(2)

                with before_col:
                    st.markdown('<div class="compare-label">Original Image</div>', unsafe_allow_html=True)
                    st.image(result["original_rgb"], use_container_width=True)

                with after_col:
                    st.markdown('<div class="compare-label">CNN Output with Stress Zones</div>', unsafe_allow_html=True)
                    st.image(result["result_rgb"], use_container_width=True)

                low, medium, high = result["percentages"]

                bar_text = "\n".join([
                    make_text_bar("High", high),
                    make_text_bar("Medium", medium),
                    make_text_bar("Low", low),
                ])

                st.markdown(
                    f"""
                    <div class="caption-bar">
                        <span style="color:{CLASS_COLORS_HEX[HIGH_CLASS]};">High</span> {high:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp;
                        <span style="color:{CLASS_COLORS_HEX[MEDIUM_CLASS]};">Medium</span> {medium:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp;
                        <span style="color:{CLASS_COLORS_HEX[LOW_CLASS]};">Low</span> {low:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Yield Intensity Bars**")
                st.code(bar_text)

                metric1, metric2, metric3, metric4 = st.columns(4)
                metric1.metric("Low Yield", f"{low:.2f}%")
                metric2.metric("Medium Yield", f"{medium:.2f}%")
                metric3.metric("High Yield", f"{high:.2f}%")
                metric4.metric("Total Patches", f"{result['total']}")

                st.markdown("**Zone Marking Basis**")
                st.write(f"- The image is split into {PATCH_SIZE} x {PATCH_SIZE} pixel patches.")
                st.write(f"- Each patch is resized to {MODEL_INPUT_SIZE[0]} x {MODEL_INPUT_SIZE[1]} pixels before CNN prediction.")
                st.write("- The CNN predicts whether each patch is Low Yield, Medium Yield, or High Yield.")
                st.write("- Neighboring patches predicted as Low Yield are grouped and marked as stress zones.")

                chart_col, table_col = st.columns([1.2, 1.0])

                with chart_col:
                    st.pyplot(result["pie_fig"])

                with table_col:
                    st.markdown("**Summary Table**")
                    st.dataframe(result["summary_table"], use_container_width=True, hide_index=True)

                if result["region_boxes"]:
                    st.markdown("**Detected Stress Zones**")
                    for box in result["region_boxes"]:
                        st.write(f"- {box['label']}: {box['cells']} connected low-yield patches")
                else:
                    st.write("No major connected low-yield zone detected.")

                st.download_button(
                    label=f"Download PDF Report for {result['name']}",
                    data=result["pdf_buffer"],
                    file_name=f"{os.path.splitext(result['name'])[0]}_CropSense_Report.pdf",
                    mime="application/pdf",
                )
    elif analyze:
        st.warning("No results were generated. Please check the uploaded images.")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">Session Summary</div>', unsafe_allow_html=True)

    if results:
        avg_low = np.mean([item["percentages"][0] for item in results])
        avg_medium = np.mean([item["percentages"][1] for item in results])
        avg_high = np.mean([item["percentages"][2] for item in results])

        st.markdown(
            f"""
            <div class="summary-row">
                <span class="chip"><span class="legend-box" style="background:{CLASS_COLORS_HEX[HIGH_CLASS]};"></span> Avg High</span>
                <span>{avg_high:.1f}%</span>
            </div>
            <div class="summary-row">
                <span class="chip"><span class="legend-box" style="background:{CLASS_COLORS_HEX[MEDIUM_CLASS]};"></span> Avg Medium</span>
                <span>{avg_medium:.1f}%</span>
            </div>
            <div class="summary-row">
                <span class="chip"><span class="legend-box" style="background:{CLASS_COLORS_HEX[LOW_CLASS]};"></span> Avg Low</span>
                <span>{avg_low:.1f}%</span>
            </div>
            <div class="summary-row">
                <span class="chip">Images Processed</span>
                <span>{len(results)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.write("Session results will appear here after analysis.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">Field Insights</div>', unsafe_allow_html=True)

    if results:
        selected = st.selectbox(
            "Select image for insights",
            options=[item["name"] for item in results],
            label_visibility="collapsed",
        )
        selected_result = next(item for item in results if item["name"] == selected)
        for recommendation in selected_result["recommendations"]:
            st.write(f"- {recommendation}")
    else:
        st.write("Recommendations will appear here after analysis.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

