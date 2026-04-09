import streamlit as st
import cv2
import numpy as np
import joblib
import tempfile
import matplotlib.pyplot as plt
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="CropSense", page_icon="🌾")

st.title("🌾 CropSense")
st.subheader("Smart Crop Yield Analyzer")

# ==============================
# LOAD MODEL
# ==============================
if not os.path.exists("crop_model.pkl"):
    st.error("❌ Model file not found (crop_model.pkl)")
    st.stop()

model = joblib.load("crop_model.pkl")

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload Farm Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    img = cv2.imread(tfile.name)

    if img is None:
        st.error("❌ Unable to read image")
        st.stop()

    # Show uploaded image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="📤 Uploaded Image", width=700)

    patch_size = 64
    h, w, _ = img.shape

    output_img = img.copy()
    counts = [0, 0, 0]  # low, medium, high

    # ==============================
    # PROCESS IMAGE
    # ==============================
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):

            patch = img[y:y+patch_size, x:x+patch_size]

            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:

                patch = patch / 255.0
                patch = patch.flatten().reshape(1, -1)

                pred = model.predict(patch)[0]
                counts[pred] += 1

                # 🎨 COLORS (BGR)
                if pred == 0:
                    color = (0, 0, 255)      # RED → LOW YIELD
                elif pred == 1:
                    color = (0, 255, 255)    # YELLOW → MEDIUM YIELD
                else:
                    color = (0, 255, 0)      # GREEN → HIGH YIELD

                cv2.rectangle(output_img, (x, y), (x+patch_size, y+patch_size), color, 2)

    total = sum(counts)

    if total == 0:
        st.warning("⚠ No valid patches found")
        st.stop()

    # ==============================
    # RESULTS
    # ==============================
    low = counts[0]/total*100
    medium = counts[1]/total*100
    high = counts[2]/total*100

    st.subheader("📊 Crop Yield Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Low Yield 🌱", f"{low:.2f}%")
    col2.metric("Medium Yield 🌾", f"{medium:.2f}%")
    col3.metric("High Yield 🌳", f"{high:.2f}%")

    # ==============================
    # PIE CHART
    # ==============================
    st.subheader("📈 Yield Distribution")

    labels = ['Low Yield', 'Medium Yield', 'High Yield']
    sizes = [low, medium, high]
    colors = ['red', 'yellow', 'green']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    ax.set_title("Crop Yield Distribution")

    st.pyplot(fig)

    # ==============================
    # SHOW OUTPUT IMAGE
    # ==============================
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    st.image(output_img, caption="🧠 CropSense Output", width=700)

    st.success("✅ Analysis Complete")