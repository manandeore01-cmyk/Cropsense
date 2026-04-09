import cv2
import numpy as np
import joblib
import os

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("crop_model.pkl")

# ==============================
# ANALYZE FUNCTION
# ==============================
def analyze_farm(image_path):
    
    # check if file exists
    if not os.path.exists(image_path):
        print("❌ Error: Image path is wrong")
        return

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: Unable to read image")
        return

    patch_size = 64
    h, w, _ = img.shape

    output_img = img.copy()
    counts = [0, 0, 0]  # low, medium, high

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):

            patch = img[y:y+patch_size, x:x+patch_size]

            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:

                patch = patch / 255.0
                patch = patch.flatten().reshape(1, -1)

                pred = model.predict(patch)[0]
                counts[pred] += 1

                # COLORING
                if pred == 0:
                    color = (0, 0, 255)      # RED → LOW
                elif pred == 1:
                    color = (0, 255, 255)    # YELLOW → MEDIUM
                else:
                    color = (0, 255, 0)      # GREEN → HIGH

                cv2.rectangle(output_img, (x, y), (x+patch_size, y+patch_size), color, 2)

    total = sum(counts)

    if total == 0:
        print("❌ No valid patches found")
        return

    # PRINT RESULT
    print("\n🌾 Farm Analysis Result:")
    print(f"Low: {counts[0]/total*100:.2f}%")
    print(f"Medium: {counts[1]/total*100:.2f}%")
    print(f"High: {counts[2]/total*100:.2f}%")

    # SAVE IMAGE
    cv2.imwrite("output_result.png", output_img)
    print("✅ Output image saved as output_result.png")


# ==============================
# YOUR IMAGE PATH (FINAL)
# ==============================
analyze_farm(r"D:\FINAL YEAR PROJECT FILES\Testing\Farm image.png")