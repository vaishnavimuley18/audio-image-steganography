from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATH
# ============================================================

DATASET_FILE = Path(
    "results/multi_cover_analysis/recommender_feature_dataset.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RECOMMENDER DATASET INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    df = pd.read_csv(DATASET_FILE)

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATA TYPES")
    print("-" * 70)

    print(df.dtypes)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")

    else:
        missing_table = pd.DataFrame({
            "missing_count": missing,
            "missing_percent": (
                missing / len(df) * 100
            ).round(2)
        })

        print(missing_table)

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CONSTANT FEATURES")
    print("-" * 70)

    constant_columns = []

    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)

    if constant_columns:

        print(
            f"Found {len(constant_columns)} constant columns:"
        )

        for column in constant_columns:
            print(f"  - {column}")

    else:

        print("No constant columns.")

    # --------------------------------------------------------
    # Unique values
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FEATURE CARDINALITY")
    print("-" * 70)

    cardinality = pd.DataFrame({
        "feature": df.columns,
        "unique_values": [
            df[column].nunique(dropna=False)
            for column in df.columns
        ],
        "dtype": [
            str(df[column].dtype)
            for column in df.columns
        ]
    })

    print(
        cardinality.to_string(index=False)
    )

    # --------------------------------------------------------
    # Scenario counts
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("SCENARIO STRUCTURE")
    print("-" * 70)

    print(
        f"Unique covers   : {df['cover_name'].nunique()}"
    )

    print(
        f"Unique payloads : {df['payload_name'].nunique()}"
    )

    print(
        f"Unique methods  : {df['method'].nunique()}"
    )

    print(
        f"Payload types   : {df['payload_type'].unique()}"
    )

    # --------------------------------------------------------
    # Method distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("METHOD DISTRIBUTION")
    print("-" * 70)

    print(
        df["method"].value_counts()
    )

    # --------------------------------------------------------
    # Utility distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("UTILITY SCORE DISTRIBUTION")
    print("-" * 70)

    print(
        df["utility_score"].describe()
    )

    # --------------------------------------------------------
    # Utility by method
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("UTILITY BY METHOD")
    print("-" * 70)

    method_utility = (
        df.groupby("method")["utility_score"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
        .round(4)
    )

    print(method_utility)

    # --------------------------------------------------------
    # Utility by payload type
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("UTILITY BY PAYLOAD TYPE")
    print("-" * 70)

    payload_utility = (
        df.groupby("payload_type")["utility_score"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
        .round(4)
    )

    print(payload_utility)

    # --------------------------------------------------------
    # Best empirical method per scenario
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("EMPIRICAL BEST METHOD")
    print("-" * 70)

    scenario_best = (
        df.loc[
            df.groupby(
                ["cover_name", "payload_name"]
            )["utility_score"].idxmax()
        ]
        [
            [
                "cover_name",
                "payload_name",
                "payload_type",
                "method",
                "utility_score",
            ]
        ]
        .copy()
    )

    print(
        scenario_best["method"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("NUMERIC FEATURE CORRELATION WITH UTILITY")
    print("-" * 70)

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    correlations = (
        numeric_df.corr()["utility_score"]
        .drop("utility_score")
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print(
        correlations.round(4)
    )

    # --------------------------------------------------------
    # Possible leakage columns
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("POTENTIAL POST-EMBEDDING / LEAKAGE FEATURES")
    print("-" * 70)

    leakage_candidates = [
        "embedding_time_seconds",
        "extraction_time_seconds",
        "integrity_percent",
        "mse",
        "psnr",
        "ssim",
        "changed_pixels",
        "changed_pixel_percentage",
        "changed_channels",
        "changed_channel_percentage",
        "status",
        "error",
        "stego_path",
        "is_success",
        "reliability_score",
        "psnr_numeric",
        "ssim_numeric",
        "mse_numeric",
        "embedding_time_numeric",
        "extraction_time_numeric",
        "psnr_score",
        "ssim_score",
        "embedding_speed_score",
        "extraction_speed_score",
        "distortion_score",
        "capacity_margin_score",
        "utility_rank",
        "is_best_candidate",
    ]

    present_leakage = [
        column
        for column in leakage_candidates
        if column in df.columns
    ]

    if present_leakage:

        for column in present_leakage:
            print(f"  - {column}")

    else:

        print("No leakage candidates found.")

    # --------------------------------------------------------
    # Proposed model features
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("PROPOSED MODEL FEATURES")
    print("-" * 70)

    proposed_features = [
        "method_encoded",

        "cover_cover_width",
        "cover_cover_height",
        "cover_cover_channels",
        "cover_cover_pixels",
        "cover_mean_intensity",
        "cover_std_intensity",
        "cover_entropy",
        "cover_edge_density",

        "payload_payload_size_bytes",
        "payload_payload_size_kb",
        "payload_payload_size_mb",
        "payload_text_length",

        "payload_duration_seconds",
        "payload_sample_rate",
        "payload_channels",
        "payload_sample_width_bytes",
        "payload_audio_frames",

        "payload_payload_width",
        "payload_payload_height",
        "payload_payload_channels",
        "payload_payload_pixels",

        "capacity_bytes",
        "required_bits",
        "capacity_fit_numeric",
        "capacity_utilization_percent",
    ]

    present_features = [
        feature
        for feature in proposed_features
        if feature in df.columns
    ]

    missing_features = [
        feature
        for feature in proposed_features
        if feature not in df.columns
    ]

    print(
        f"\nAvailable proposed features: "
        f"{len(present_features)}"
    )

    for feature in present_features:
        print(f"  ✓ {feature}")

    if missing_features:

        print(
            f"\nMissing proposed features: "
            f"{len(missing_features)}"
        )

        for feature in missing_features:
            print(f"  ✗ {feature}")

    # --------------------------------------------------------
    # Train/test groups
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("GROUPED TRAIN / TEST SPLIT")
    print("-" * 70)

    covers = sorted(
        df["cover_name"].unique()
    )

    print(
        f"Total cover groups: {len(covers)}"
    )

    # Deterministic split
    train_covers = covers[:8]
    test_covers = covers[8:]

    print("\nTraining covers:")

    for cover in train_covers:
        print(f"  {cover}")

    print("\nTesting covers:")

    for cover in test_covers:
        print(f"  {cover}")

    train_df = df[
        df["cover_name"].isin(train_covers)
    ]

    test_df = df[
        df["cover_name"].isin(test_covers)
    ]

    print(
        f"\nTraining rows: {len(train_df)}"
    )

    print(
        f"Testing rows : {len(test_df)}"
    )

    # --------------------------------------------------------
    # Check candidate balance in each split
    # --------------------------------------------------------

    print("\nTraining method distribution:")

    print(
        train_df["method"].value_counts()
    )

    print("\nTesting method distribution:")

    print(
        test_df["method"].value_counts()
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()