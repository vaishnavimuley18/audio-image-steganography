from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

COVER_DIR = Path("output/stego/multi_cover_test")

OUTPUT_DIR = Path("results/multi_cover_analysis")
OUTPUT_FILE = OUTPUT_DIR / "processed_cover_diversity.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def calculate_entropy(gray_image):
    histogram = cv2.calcHist(
        [gray_image],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    histogram = histogram / histogram.sum()
    histogram = histogram[histogram > 0]

    return float(-np.sum(histogram * np.log2(histogram)))


def calculate_edge_density(gray_image):
    edges = cv2.Canny(gray_image, 100, 200)
    return float(np.mean(edges > 0))


def analyze_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    return {
        "cover_name": image_path.name,
        "width": width,
        "height": height,
        "channels": image.shape[2],
        "pixels": width * height,
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "entropy": calculate_entropy(gray),
        "edge_density": calculate_edge_density(gray),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # Search recursively because processed covers may be
    # stored inside subdirectories.
    image_files = sorted(
        list(COVER_DIR.rglob("*.png")) +
        list(COVER_DIR.rglob("*.jpg")) +
        list(COVER_DIR.rglob("*.jpeg"))
    )

    # Remove stego images and extracted payloads.
    image_files = [
        path for path in image_files
        if "stego" not in path.name.lower()
        and "extracted" not in str(path).lower()
    ]

    print("=" * 70)
    print("PROCESSED COVER DIVERSITY AUDIT")
    print("=" * 70)

    print(f"\nSearch directory: {COVER_DIR}")
    print(f"Images found: {len(image_files)}")

    if not image_files:
        print("\nNo processed cover images found.")
        print("We will locate their actual directory in the next step.")
        return

    records = []

    for image_path in image_files:

        print(f"Processing: {image_path}")

        try:
            records.append(analyze_image(image_path))

        except Exception as e:
            print(f"  ERROR: {e}")

    df = pd.DataFrame(records)

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(OUTPUT_FILE, index=False)

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESSED COVER FEATURES")
    print("=" * 70)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    numeric_columns = [
        "mean_intensity",
        "std_intensity",
        "entropy",
        "edge_density"
    ]

    print("\n" + "=" * 70)
    print("DIVERSITY SUMMARY")
    print("=" * 70)

    summary = df[numeric_columns].agg(["min", "max", "mean", "std"]).T

    print(
        summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ========================================================
    # RANGES
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE RANGES")
    print("=" * 70)

    for column in numeric_columns:

        minimum = df[column].min()
        maximum = df[column].max()

        print(
            f"{column:20s}: "
            f"{minimum:.4f} → {maximum:.4f} "
            f"(range = {maximum - minimum:.4f})"
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()