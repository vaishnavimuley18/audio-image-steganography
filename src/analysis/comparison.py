"""
Traditional Steganography Method Comparison

Reads:
    results/experimental_results.csv

Generates:
    results/comparison/method_summary.csv
    results/comparison/payload_summary.csv
    results/comparison/successful_experiments.csv
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "results" / "experimental_results.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"

METHOD_SUMMARY_FILE = OUTPUT_DIR / "method_summary.csv"
PAYLOAD_SUMMARY_FILE = OUTPUT_DIR / "payload_summary.csv"
SUCCESSFUL_FILE = OUTPUT_DIR / "successful_experiments.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_results():
    """Load experimental results from CSV."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Experimental results file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("TRADITIONAL STEGANOGRAPHY COMPARISON")
    print("=" * 70)

    print(f"\nInput file : {INPUT_FILE}")
    print(f"Experiments: {len(df)}")

    return df


# ============================================================
# METHOD SUMMARY
# ============================================================

def create_method_summary(df):
    """
    Create one summary row for each method.

    Only successful experiments are used for numerical
    quality/time averages.
    """

    methods = ["LSB", "DCT", "DWT"]

    rows = []

    for method in methods:

        method_df = df[df["method"] == method]

        successful = method_df[
            method_df["status"] == "SUCCESS"
        ]

        rows.append({
            "method": method,

            "total_experiments":
                len(method_df),

            "successful_experiments":
                len(successful),

            "capacity_failures":
                int(
                    (method_df["status"] ==
                     "INSUFFICIENT_CAPACITY").sum()
                ),

            "integrity_failures":
                int(
                    (method_df["status"] ==
                     "INTEGRITY_FAILED").sum()
                ),

            "capacity_bytes":
                method_df["capacity_bytes"].iloc[0],

            "capacity_megabytes":
                method_df["capacity_bytes"].iloc[0]
                / (1024 * 1024),

            "average_psnr_db":
                successful["psnr"].mean(),

            "average_ssim":
                successful["ssim"].mean(),

            "average_mse":
                successful["mse"].mean(),

            "average_embedding_time_seconds":
                successful["embedding_time_seconds"].mean(),

            "average_extraction_time_seconds":
                successful["extraction_time_seconds"].mean(),

            "average_integrity_percent":
                successful["integrity_percent"].mean(),

            "successful_experiment_percent":
                (
                    len(successful)
                    / len(method_df)
                    * 100
                )
            if len(method_df) > 0 else 0
        })

    summary = pd.DataFrame(rows)

    return summary


# ============================================================
# PAYLOAD SUMMARY
# ============================================================

def create_payload_summary(df):
    """
    Compare methods for each payload type.
    """

    payload_types = ["TEXT", "AUDIO", "IMAGE"]

    rows = []

    for payload_type in payload_types:

        payload_df = df[
            df["payload_type"] == payload_type
        ]

        for _, row in payload_df.iterrows():

            rows.append({
                "payload_type":
                    row["payload_type"],

                "method":
                    row["method"],

                "payload_size_bytes":
                    row["payload_size_bytes"],

                "total_payload_size_bytes":
                    row["total_payload_size_bytes"],

                "capacity_bytes":
                    row["capacity_bytes"],

                "capacity_utilization_percent":
                    row["capacity_utilization_percent"],

                "capacity_fit":
                    row["capacity_fit"],

                "embedding_time_seconds":
                    row["embedding_time_seconds"],

                "extraction_time_seconds":
                    row["extraction_time_seconds"],

                "integrity_percent":
                    row["integrity_percent"],

                "mse":
                    row["mse"],

                "psnr":
                    row["psnr"],

                "ssim":
                    row["ssim"],

                "status":
                    row["status"]
            })

    return pd.DataFrame(rows)


# ============================================================
# SUCCESSFUL EXPERIMENTS
# ============================================================

def create_successful_experiments(df):
    """Keep only completely successful experiments."""

    return df[
        df["status"] == "SUCCESS"
    ].copy()


# ============================================================
# PRINT METHOD SUMMARY
# ============================================================

def print_method_summary(summary):

    print("\n" + "=" * 70)
    print("METHOD SUMMARY")
    print("=" * 70)

    display_columns = [
        "method",
        "capacity_bytes",
        "capacity_megabytes",
        "successful_experiments",
        "capacity_failures",
        "integrity_failures",
        "average_psnr_db",
        "average_ssim",
        "average_embedding_time_seconds",
        "average_extraction_time_seconds"
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# PRINT PAYLOAD COMPARISON
# ============================================================

def print_payload_comparison(df):

    print("\n" + "=" * 70)
    print("PAYLOAD-WISE COMPARISON")
    print("=" * 70)

    for payload_type in ["TEXT", "AUDIO", "IMAGE"]:

        print(f"\n--- {payload_type} ---")

        payload_df = df[
            df["payload_type"] == payload_type
        ]

        columns = [
            "method",
            "capacity_fit",
            "status",
            "integrity_percent",
            "psnr",
            "ssim"
        ]

        print(
            payload_df[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    method_summary,
    payload_summary,
    successful
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    method_summary.to_csv(
        METHOD_SUMMARY_FILE,
        index=False
    )

    payload_summary.to_csv(
        PAYLOAD_SUMMARY_FILE,
        index=False
    )

    successful.to_csv(
        SUCCESSFUL_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)

    print(f"\n✓ {METHOD_SUMMARY_FILE}")
    print(f"✓ {PAYLOAD_SUMMARY_FILE}")
    print(f"✓ {SUCCESSFUL_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    # Load experimental dataset
    df = load_results()

    # Create summaries
    method_summary = create_method_summary(df)

    payload_summary = create_payload_summary(df)

    successful = create_successful_experiments(df)

    # Display results
    print_method_summary(method_summary)

    print_payload_comparison(df)

    # Save results
    save_results(
        method_summary,
        payload_summary,
        successful
    )

    print("\n" + "=" * 70)
    print("COMPARISON ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()