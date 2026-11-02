from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

COVER_PATH = Path("data/images/processed/cover_processed.png")
STEGO_PATH = Path("output/stego/lsb_stego.png")

OUTPUT_DIR = Path("output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD IMAGES
# ============================================================

cover = np.array(Image.open(COVER_PATH).convert("RGB"))
stego = np.array(Image.open(STEGO_PATH).convert("RGB"))

print("=" * 60)
print("LSB VISUAL & HISTOGRAM ANALYSIS")
print("=" * 60)

print("\nINPUT IMAGES")
print("-" * 60)
print(f"Cover image : {COVER_PATH}")
print(f"Stego image : {STEGO_PATH}")
print(f"Image shape : {cover.shape}")


# ============================================================
# 1. COVER VS STEGO IMAGE
# ============================================================

print("\n1. IMAGE COMPARISON")
print("-" * 60)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cover)
plt.title("Original Cover Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(stego)
plt.title("LSB Stego Image")
plt.axis("off")

plt.tight_layout()

comparison_path = OUTPUT_DIR / "lsb_cover_vs_stego.png"
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved : {comparison_path}")


# ============================================================
# 2. RGB HISTOGRAM COMPARISON
# ============================================================

print("\n2. RGB HISTOGRAM ANALYSIS")
print("-" * 60)

channel_names = ["Red", "Green", "Blue"]

plt.figure(figsize=(14, 10))

for i, channel in enumerate(channel_names):

    plt.subplot(3, 1, i + 1)

    plt.hist(
        cover[:, :, i].ravel(),
        bins=256,
        range=(0, 256),
        alpha=0.5,
        label="Cover"
    )

    plt.hist(
        stego[:, :, i].ravel(),
        bins=256,
        range=(0, 256),
        alpha=0.5,
        label="Stego"
    )

    plt.title(f"{channel} Channel Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.2)

plt.tight_layout()

histogram_path = OUTPUT_DIR / "lsb_rgb_histogram.png"
plt.savefig(histogram_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved : {histogram_path}")


# ============================================================
# 3. AMPLIFIED DIFFERENCE IMAGE
# ============================================================

print("\n3. DIFFERENCE VISUALIZATION")
print("-" * 60)

difference = np.abs(
    cover.astype(np.int16) - stego.astype(np.int16)
)

# Amplify small LSB changes
amplified_difference = np.clip(difference * 50, 0, 255).astype(np.uint8)

plt.figure(figsize=(8, 8))
plt.imshow(amplified_difference)
plt.title("Amplified Difference (×50)")
plt.axis("off")

difference_path = OUTPUT_DIR / "lsb_difference_amplified.png"
plt.savefig(difference_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved : {difference_path}")


# ============================================================
# 4. LSB PLANE VISUALIZATION
# ============================================================

print("\n4. LSB PLANE VISUALIZATION")
print("-" * 60)

cover_lsb = cover & 1
stego_lsb = stego & 1

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cover_lsb[:, :, 0], cmap="gray")
plt.title("Cover - Red LSB")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(cover_lsb[:, :, 1], cmap="gray")
plt.title("Cover - Green LSB")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(cover_lsb[:, :, 2], cmap="gray")
plt.title("Cover - Blue LSB")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(stego_lsb[:, :, 0], cmap="gray")
plt.title("Stego - Red LSB")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(stego_lsb[:, :, 1], cmap="gray")
plt.title("Stego - Green LSB")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(stego_lsb[:, :, 2], cmap="gray")
plt.title("Stego - Blue LSB")
plt.axis("off")

plt.tight_layout()

lsb_path = OUTPUT_DIR / "lsb_plane_visualization.png"
plt.savefig(lsb_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved : {lsb_path}")


# ============================================================
# 5. NUMERICAL LSB CHANGE ANALYSIS
# ============================================================

print("\n5. LSB CHANGE ANALYSIS")
print("-" * 60)

cover_lsb_flat = cover & 1
stego_lsb_flat = stego & 1

lsb_changed = np.sum(cover_lsb_flat != stego_lsb_flat)
total_channels = cover.size

lsb_change_percentage = (
    lsb_changed / total_channels
) * 100

print(f"Total channels       : {total_channels:,}")
print(f"Changed LSB channels : {lsb_changed:,}")
print(f"LSB change           : {lsb_change_percentage:.4f}%")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("VISUAL & HISTOGRAM ANALYSIS COMPLETE ✅")
print("=" * 60)

print("\nGenerated files:")
print(f"1. {comparison_path}")
print(f"2. {histogram_path}")
print(f"3. {difference_path}")
print(f"4. {lsb_path}")

print("\nRESULT: LSB VISUAL ANALYSIS SUCCESSFUL ✅")