"""
DWT Visual Analysis

Generates:
1. Cover vs Stego comparison
2. Grayscale histogram comparison
3. Amplified difference image
4. DWT coefficient visualization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt


# ============================================================
# CONFIGURATION
# ============================================================

COVER_IMAGE = "data/images/processed/cover_processed.png"
STEGO_IMAGE = "output/stego/dwt_test.png"

OUTPUT_DIR = "output/analysis"

WAVELET = "haar"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD IMAGES
# ============================================================

cover = cv2.imread(COVER_IMAGE, cv2.IMREAD_GRAYSCALE)
stego = cv2.imread(STEGO_IMAGE, cv2.IMREAD_GRAYSCALE)

if cover is None:
    raise FileNotFoundError(f"Cover image not found: {COVER_IMAGE}")

if stego is None:
    raise FileNotFoundError(f"Stego image not found: {STEGO_IMAGE}")

if cover.shape != stego.shape:
    raise ValueError(
        f"Image dimensions do not match: "
        f"Cover={cover.shape}, Stego={stego.shape}"
    )


print("=" * 70)
print("DWT VISUAL ANALYSIS")
print("=" * 70)

print(f"Cover image : {COVER_IMAGE}")
print(f"Stego image : {STEGO_IMAGE}")
print(f"Image size  : {cover.shape[1]} x {cover.shape[0]}")


# ============================================================
# 1. COVER VS STEGO IMAGE
# ============================================================

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(cover, cmap="gray")
plt.title("Original Cover Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(stego, cmap="gray")
plt.title("DWT Stego Image")
plt.axis("off")

plt.tight_layout()

comparison_path = os.path.join(
    OUTPUT_DIR,
    "dwt_cover_vs_stego.png"
)

plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
plt.close()

print("\n[1] Cover vs Stego comparison")
print(f"    Saved : {comparison_path}")


# ============================================================
# 2. GRAYSCALE HISTOGRAM COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

plt.hist(
    cover.ravel(),
    bins=256,
    range=(0, 256),
    alpha=0.5,
    label="Cover"
)

plt.hist(
    stego.ravel(),
    bins=256,
    range=(0, 256),
    alpha=0.5,
    label="DWT Stego"
)

plt.title("DWT Cover vs Stego Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()

histogram_path = os.path.join(
    OUTPUT_DIR,
    "dwt_grayscale_histogram.png"
)

plt.savefig(histogram_path, dpi=200, bbox_inches="tight")
plt.close()

print("\n[2] Grayscale histogram comparison")
print(f"    Saved : {histogram_path}")


# ============================================================
# 3. DIFFERENCE IMAGE
# ============================================================

difference = cv2.absdiff(cover, stego)

# Amplify differences for visibility
amplification_factor = 5

difference_amplified = np.clip(
    difference.astype(np.float32) * amplification_factor,
    0,
    255
).astype(np.uint8)

difference_path = os.path.join(
    OUTPUT_DIR,
    "dwt_difference_amplified.png"
)

cv2.imwrite(
    difference_path,
    difference_amplified
)

print("\n[3] Amplified difference image")
print(f"    Amplification factor : {amplification_factor}x")
print(f"    Saved : {difference_path}")


# ============================================================
# 4. DWT COEFFICIENT VISUALIZATION
# ============================================================

cover_float = cover.astype(np.float32)
stego_float = stego.astype(np.float32)

cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(
    cover_float,
    WAVELET
)

stego_LL, (stego_LH, stego_HL, stego_HH) = pywt.dwt2(
    stego_float,
    WAVELET
)


# Function to normalize coefficient images
def normalize_coefficients(coefficients):
    coefficients = np.abs(coefficients)

    min_value = coefficients.min()
    max_value = coefficients.max()

    if max_value == min_value:
        return np.zeros_like(coefficients)

    normalized = (
        (coefficients - min_value)
        / (max_value - min_value)
        * 255
    )

    return normalized.astype(np.uint8)


# We focus on LH and HL because those are the
# DWT subbands used by the current embedding method.

cover_LH_visual = normalize_coefficients(cover_LH)
cover_HL_visual = normalize_coefficients(cover_HL)

stego_LH_visual = normalize_coefficients(stego_LH)
stego_HL_visual = normalize_coefficients(stego_HL)


plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(cover_LH_visual, cmap="gray")
plt.title("Cover - LH Subband")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(stego_LH_visual, cmap="gray")
plt.title("DWT Stego - LH Subband")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cover_HL_visual, cmap="gray")
plt.title("Cover - HL Subband")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(stego_HL_visual, cmap="gray")
plt.title("DWT Stego - HL Subband")
plt.axis("off")

plt.tight_layout()

coefficients_path = os.path.join(
    OUTPUT_DIR,
    "dwt_coefficient_visualization.png"
)

plt.savefig(
    coefficients_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\n[4] DWT coefficient visualization")
print(f"    Saved : {coefficients_path}")


# ============================================================
# 5. DWT COEFFICIENT DIFFERENCE
# ============================================================

lh_difference = np.abs(cover_LH - stego_LH)
hl_difference = np.abs(cover_HL - stego_HL)

lh_visual = normalize_coefficients(lh_difference)
hl_visual = normalize_coefficients(hl_difference)


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(lh_visual, cmap="gray")
plt.title("LH Coefficient Difference")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hl_visual, cmap="gray")
plt.title("HL Coefficient Difference")
plt.axis("off")

plt.tight_layout()

coefficient_difference_path = os.path.join(
    OUTPUT_DIR,
    "dwt_coefficient_difference.png"
)

plt.savefig(
    coefficient_difference_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\n[5] DWT coefficient difference")
print(f"    Saved : {coefficient_difference_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DWT VISUAL ANALYSIS COMPLETE ✅")
print("=" * 70)

print("\nGenerated files:")

print(f"1. {comparison_path}")
print(f"2. {histogram_path}")
print(f"3. {difference_path}")
print(f"4. {coefficients_path}")
print(f"5. {coefficient_difference_path}")

print("\n" + "=" * 70)