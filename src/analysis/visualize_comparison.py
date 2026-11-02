"""
Visual Comparison of Traditional Steganography Methods

Reads:
    results/experimental_results.csv

Generates:
    results/comparison/capacity_comparison.png
    results/comparison/psnr_comparison.png
    results/comparison/ssim_comparison.png
    results/comparison/execution_time_comparison.png
    results/comparison/payload_success_matrix.png
    results/comparison/distortion_comparison.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "results" / "experimental_results.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load experimental results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Experimental results not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return df


# ============================================================
# 1. CAPACITY COMPARISON
# ============================================================

def plot_capacity(df):

    summary = (
        df.groupby("method")["capacity_bytes"]
        .first()
        .reset_index()
    )

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        summary["method"],
        summary["capacity_bytes"]
    )

    plt.title("Steganography Method Capacity")
    plt.xlabel("Method")
    plt.ylabel("Capacity (Bytes)")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(
        bars,
        summary["capacity_bytes"]
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output = OUTPUT_DIR / "capacity_comparison.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Capacity plot: {output}")


# ============================================================
# 2. PSNR COMPARISON
# ============================================================

def plot_psnr(df):

    successful = df[
        df["status"] == "SUCCESS"
    ].copy()

    plt.figure(figsize=(10, 6))

    for method in ["LSB", "DCT", "DWT"]:

        method_df = successful[
            successful["method"] == method
        ]

        if len(method_df) == 0:
            continue

        plt.plot(
            method_df["payload_type"],
            method_df["psnr"],
            marker="o",
            label=method
        )

    plt.title("PSNR Comparison for Successful Experiments")
    plt.xlabel("Payload Type")
    plt.ylabel("PSNR (dB)")

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "psnr_comparison.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ PSNR plot: {output}")


# ============================================================
# 3. SSIM COMPARISON
# ============================================================

def plot_ssim(df):

    successful = df[
        df["status"] == "SUCCESS"
    ].copy()

    plt.figure(figsize=(10, 6))

    for method in ["LSB", "DCT", "DWT"]:

        method_df = successful[
            successful["method"] == method
        ]

        if len(method_df) == 0:
            continue

        plt.plot(
            method_df["payload_type"],
            method_df["ssim"],
            marker="o",
            label=method
        )

    plt.title("SSIM Comparison for Successful Experiments")
    plt.xlabel("Payload Type")
    plt.ylabel("SSIM")

    plt.ylim(
        0.99,
        1.001
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "ssim_comparison.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ SSIM plot: {output}")


# ============================================================
# 4. EXECUTION TIME
# ============================================================

def plot_execution_time(df):

    successful = df[
        df["status"] == "SUCCESS"
    ].copy()

    summary = (
        successful
        .groupby("method")[
            [
                "embedding_time_seconds",
                "extraction_time_seconds"
            ]
        ]
        .mean()
        .reset_index()
    )

    x = range(len(summary))
    width = 0.35

    plt.figure(figsize=(9, 6))

    embedding = [
        value
        for value in summary[
            "embedding_time_seconds"
        ]
    ]

    extraction = [
        value
        for value in summary[
            "extraction_time_seconds"
        ]
    ]

    bars1 = plt.bar(
        [i - width / 2 for i in x],
        embedding,
        width,
        label="Embedding"
    )

    bars2 = plt.bar(
        [i + width / 2 for i in x],
        extraction,
        width,
        label="Extraction"
    )

    plt.xticks(
        list(x),
        summary["method"]
    )

    plt.title(
        "Average Execution Time for Successful Experiments"
    )

    plt.xlabel("Method")
    plt.ylabel("Time (seconds)")

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bars in [bars1, bars2]:

        for bar in bars:

            value = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.tight_layout()

    output = OUTPUT_DIR / "execution_time_comparison.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Execution-time plot: {output}")


# ============================================================
# 5. PAYLOAD SUCCESS MATRIX
# ============================================================

def plot_success_matrix(df):

    payload_types = [
        "TEXT",
        "AUDIO",
        "IMAGE"
    ]

    methods = [
        "LSB",
        "DCT",
        "DWT"
    ]

    matrix = []

    for payload in payload_types:

        row = []

        for method in methods:

            result = df[
                (df["payload_type"] == payload)
                &
                (df["method"] == method)
            ]

            if len(result) == 0:
                row.append("N/A")

            else:

                status = result.iloc[0]["status"]

                if status == "SUCCESS":
                    row.append("SUCCESS")

                elif status == "INSUFFICIENT_CAPACITY":
                    row.append("NO CAPACITY")

                elif status == "INTEGRITY_FAILED":
                    row.append("INTEGRITY FAIL")

                else:
                    row.append("FAILED")

        matrix.append(row)

    plt.figure(figsize=(9, 5))

    # Numerical representation only for positioning text.
    numeric = []

    for row in matrix:

        numeric_row = []

        for value in row:

            if value == "SUCCESS":
                numeric_row.append(2)

            elif value == "NO CAPACITY":
                numeric_row.append(1)

            elif value == "INTEGRITY FAIL":
                numeric_row.append(0)

            else:
                numeric_row.append(0)

        numeric.append(numeric_row)

    plt.imshow(
        numeric,
        aspect="auto"
    )

    plt.xticks(
        range(len(methods)),
        methods
    )

    plt.yticks(
        range(len(payload_types)),
        payload_types
    )

    plt.xlabel("Method")
    plt.ylabel("Payload Type")

    plt.title(
        "Traditional Steganography Payload Compatibility"
    )

    for i in range(len(payload_types)):

        for j in range(len(methods)):

            plt.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center",
                fontsize=8
            )

    plt.colorbar(
        label="Result Category"
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "payload_success_matrix.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Payload matrix: {output}")


# ============================================================
# 6. DISTORTION COMPARISON
# ============================================================

def plot_distortion(df):

    successful = df[
        df["status"] == "SUCCESS"
    ].copy()

    summary = (
        successful
        .groupby("method")[
            [
                "mse",
                "changed_pixel_percentage"
            ]
        ]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        summary["method"],
        summary["mse"],
        marker="o",
        label="MSE"
    )

    plt.title(
        "Average Image Distortion"
    )

    plt.xlabel("Method")
    plt.ylabel("Mean Squared Error")

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "distortion_comparison.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"✓ Distortion plot: {output}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRADITIONAL STEGANOGRAPHY VISUAL ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(f"\nExperiments loaded: {len(df)}")

    print("\nGenerating plots...\n")

    plot_capacity(df)

    plot_psnr(df)

    plot_ssim(df)

    plot_execution_time(df)

    plot_success_matrix(df)

    plot_distortion(df)

    print("\n" + "=" * 70)
    print("VISUAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()