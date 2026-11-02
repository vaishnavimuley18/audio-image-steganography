from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

COVER_DIR = Path("data/images/covers")
OUTPUT_DIR = Path("results/multi_cover_analysis")
OUTPUT_FILE = OUTPUT_DIR / "cover_diversity.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def calculate_entropy(gray_image):
    """Calculate grayscale image entropy."""
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
    """Calculate percentage of edge pixels using Canny."""
    edges = cv2.Canny(gray_image, 100, 200)

    return float(np.mean(edges > 0))


def analyze_cover(image_path):
    """Extract diversity-related features from one cover."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    channels = image.shape[2] if len(image.shape) == 3 else 1

    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    entropy = calculate_entropy(gray)
    edge_density = calculate_edge_density(gray)

    return {
        "cover_name": image_path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "pixels": width * height,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "entropy": entropy,
        "edge_density": edge_density,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    cover_files = sorted(COVER_DIR.glob("cover_*.jpg"))

    if not cover_files:
        print("ERROR: No cover images found.")
        print(f"Expected covers inside: {COVER_DIR}")
        return

    print("=" * 70)
    print("COVER DIVERSITY AUDIT")
    print("=" * 70)

    print(f"\nFound {len(cover_files)} covers.")

    records = []

    for cover_path in cover_files:

        print(f"Processing: {cover_path.name}")

        try:
            features = analyze_cover(cover_path)
            records.append(features)

        except Exception as e:
            print(f"  ERROR: {e}")

    df = pd.DataFrame(records)

    # ========================================================
    # SAVE DATASET
    # ========================================================

    df.to_csv(OUTPUT_FILE, index=False)

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("COVER FEATURES")
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
        "edge_density",
    ]

    print("\n" + "=" * 70)
    print("DIVERSITY SUMMARY")
    print("=" * 70)

    summary = df[numeric_columns].agg(
        ["min", "max", "mean", "std"]
    ).T

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
        difference = maximum - minimum

        print(
            f"{column:20s}: "
            f"{minimum:.4f} → {maximum:.4f} "
            f"(range = {difference:.4f})"
        )

    # ========================================================
    # SORTED VIEWS
    # ========================================================

    print("\n" + "=" * 70)
    print("LOWEST / HIGHEST ENTROPY")
    print("=" * 70)

    entropy_sorted = df.sort_values("entropy")

    print(
        entropy_sorted[
            ["cover_name", "entropy"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n" + "=" * 70)
    print("LOWEST / HIGHEST EDGE DENSITY")
    print("=" * 70)

    edge_sorted = df.sort_values("edge_density")

    print(
        edge_sorted[
            ["cover_name", "edge_density"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"Dataset saved to:")
    print(OUTPUT_FILE)

    print("\nUse this output to decide which additional cover")
    print("categories are needed for the expanded experiment.")


if __name__ == "__main__":
    main()