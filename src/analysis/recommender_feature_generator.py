from pathlib import Path
import pandas as pd
import numpy as np

from src.analysis.image_features import extract_image_features
from src.analysis.payload_features import extract_payload_features


# ============================================================
# PATHS
# ============================================================

RESULTS_FILE = Path("results/multi_cover_results.csv")
OUTPUT_DIR = Path("results/multi_cover_analysis")

OUTPUT_FILE = OUTPUT_DIR / "recommender_feature_dataset.csv"

CANDIDATE_FILE = OUTPUT_DIR / "candidate_ml_dataset.csv"


# ============================================================
# METHOD ENCODING
# ============================================================

METHOD_ENCODING = {
    "LSB": 0,
    "DCT": 1,
    "DWT": 2,
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RECOMMENDER FEATURE DATASET GENERATOR")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # LOAD EXPERIMENT RESULTS
    # ========================================================

    df = pd.read_csv(RESULTS_FILE)

    print(f"\nLoaded results: {df.shape}")

    # ========================================================
    # LOAD CANDIDATE DATASET
    # ========================================================

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Candidate ML dataset not found:\n{CANDIDATE_FILE}"
        )

    candidate_df = pd.read_csv(CANDIDATE_FILE)

    print(f"Candidate ML dataset: {candidate_df.shape}")

    required_candidate_columns = [
        "cover_name",
        "payload_name",
        "method",
        "utility_score",
    ]

    missing_candidate_columns = [
        col
        for col in required_candidate_columns
        if col not in candidate_df.columns
    ]

    if missing_candidate_columns:
        raise ValueError(
            "Candidate dataset is missing required columns: "
            f"{missing_candidate_columns}"
        )

    # ========================================================
    # EXTRACT COVER FEATURES
    # ========================================================

    cover_features_cache = {}

    unique_covers = sorted(df["cover_name"].unique())

    print(f"\nExtracting features from {len(unique_covers)} covers...")

    for cover_name in unique_covers:

        # Example result filename:
        # cover_01_processed.png
        #
        # Actual source filename:
        # cover_01.jpg

        base_cover_name = (
            Path(cover_name)
            .stem
            .replace("_processed", "")
        )

        cover_path = (
            Path("data/images/covers")
            / f"{base_cover_name}.jpg"
        )

        if not cover_path.exists():

            print(
                f"WARNING: Cover not found: {cover_path}"
            )

            continue

        print(f"  Processing: {cover_name}")

        features = extract_image_features(
            str(cover_path)
        )

        cover_features_cache[cover_name] = features

    # ========================================================
    # EXTRACT PAYLOAD FEATURES
    # ========================================================

    payload_features_cache = {}

    unique_payloads = sorted(df["payload_name"].unique())

    print(f"\nExtracting features from {len(unique_payloads)} payloads...")

    for payload_name in unique_payloads:

        payload_path = (
            Path("data/payloads/variants")
            / payload_name
        )

        if not payload_path.exists():

            print(
                f"WARNING: Payload not found: {payload_path}"
            )

            continue

        print(f"  Processing: {payload_name}")

        features = extract_payload_features(
            str(payload_path)
        )

        payload_features_cache[payload_name] = features

    # ========================================================
    # BUILD PRE-EMBEDDING CANDIDATE FEATURES
    # ========================================================

    rows = []

    print("\nBuilding pre-embedding candidate rows...")

    for _, row in df.iterrows():

        cover_name = row["cover_name"]
        payload_name = row["payload_name"]
        method = row["method"]

        cover_features = cover_features_cache.get(
            cover_name,
            {}
        )

        payload_features = payload_features_cache.get(
            payload_name,
            {}
        )

        candidate = {}

        # ----------------------------------------------------
        # Scenario identifiers
        # ----------------------------------------------------

        candidate["cover_name"] = cover_name
        candidate["payload_name"] = payload_name
        candidate["payload_type"] = row["payload_type"]
        candidate["method"] = method

        # ----------------------------------------------------
        # Method encoding
        # ----------------------------------------------------

        candidate["method_encoded"] = (
            METHOD_ENCODING[method]
        )

        # ----------------------------------------------------
        # Cover features
        # ----------------------------------------------------

        for key, value in cover_features.items():

            candidate[f"cover_{key}"] = value

        # ----------------------------------------------------
        # Payload features
        # ----------------------------------------------------

        for key, value in payload_features.items():

            candidate[f"payload_{key}"] = value

        # ----------------------------------------------------
        # Capacity information
        # ----------------------------------------------------

        candidate["payload_size_bytes"] = (
            row["payload_size_bytes"]
        )

        candidate["capacity_bytes"] = (
            row["capacity_bytes"]
        )

        candidate["required_bits"] = (
            row["required_bits"]
        )

        candidate["capacity_fit"] = (
            row["capacity_fit"]
        )

        candidate["capacity_fit_numeric"] = (
            1 if row["capacity_fit"] else 0
        )

        candidate["capacity_utilization_percent"] = (
            row["capacity_utilization_percent"]
        )

        rows.append(candidate)

    feature_df = pd.DataFrame(rows)

    # ========================================================
    # REMOVE DUPLICATE FEATURE COLUMNS
    # ========================================================

    feature_df = feature_df.loc[
        :,
        ~feature_df.columns.duplicated()
    ]

    # ========================================================
    # EXTRACT UTILITY TARGET
    # ========================================================

    target_df = candidate_df[
        [
            "cover_name",
            "payload_name",
            "method",
            "utility_score",
        ]
    ].copy()

    # Make sure one target exists per candidate
    duplicate_targets = target_df.duplicated(
        subset=[
            "cover_name",
            "payload_name",
            "method",
        ]
    )

    if duplicate_targets.any():

        raise ValueError(
            "Duplicate utility targets detected for "
            "(cover, payload, method)."
        )

    # ========================================================
    # MERGE TARGET
    # ========================================================

    feature_df = feature_df.merge(
        target_df,
        on=[
            "cover_name",
            "payload_name",
            "method",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # REMOVE ANY UNEXPECTED DUPLICATE COLUMNS
    # ========================================================

    feature_df = feature_df.loc[
        :,
        ~feature_df.columns.duplicated()
    ]

    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    identifier_columns = [
        "cover_name",
        "payload_name",
        "payload_type",
        "method",
        "method_encoded",
    ]

    capacity_columns = [
        "payload_size_bytes",
        "capacity_bytes",
        "required_bits",
        "capacity_fit",
        "capacity_fit_numeric",
        "capacity_utilization_percent",
    ]

    target_column = [
        "utility_score"
    ]

    other_columns = [
        col
        for col in feature_df.columns
        if col not in (
            identifier_columns
            + capacity_columns
            + target_column
        )
    ]

    feature_df = feature_df[
        identifier_columns
        + other_columns
        + capacity_columns
        + target_column
    ]

    # ========================================================
    # SAVE
    # ========================================================

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET CREATED")
    print("=" * 70)

    print(f"\nFile: {OUTPUT_FILE}")
    print(f"Shape: {feature_df.shape}")

    print("\nColumns:")

    for i, column in enumerate(
        feature_df.columns,
        start=1
    ):
        print(
            f"{i:2}. {column}"
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = feature_df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        print("No missing values.")

    else:

        print(missing)

    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    print("\n" + "-" * 70)
    print("UTILITY SCORE STATISTICS")
    print("-" * 70)

    print(
        feature_df["utility_score"].describe()
    )

    # ========================================================
    # UTILITY BY METHOD
    # ========================================================

    print("\n" + "-" * 70)
    print("UTILITY BY METHOD")
    print("-" * 70)

    method_summary = (
        feature_df
        .groupby("method")["utility_score"]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max",
            ]
        )
        .round(4)
    )

    print(method_summary)

    # ========================================================
    # METHOD DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 70)
    print("METHOD DISTRIBUTION")
    print("-" * 70)

    print(
        feature_df["method"].value_counts()
    )

    # ========================================================
    # CAPACITY FIT
    # ========================================================

    print("\n" + "-" * 70)
    print("CAPACITY FIT")
    print("-" * 70)

    capacity_summary = (
        feature_df
        .groupby("method")[
            "capacity_fit_numeric"
        ]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
        .round(4)
    )

    print(capacity_summary)

    # ========================================================
    # TARGET COMPLETENESS
    # ========================================================

    print("\n" + "-" * 70)
    print("TARGET COMPLETENESS")
    print("-" * 70)

    target_missing = (
        feature_df["utility_score"]
        .isnull()
        .sum()
    )

    print(
        f"Missing utility scores : {target_missing}"
    )

    print(
        f"Total candidates       : {len(feature_df)}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n" + "-" * 70)
    print("FINAL VALIDATION")
    print("-" * 70)

    expected_rows = 270

    print(
        f"Expected rows : {expected_rows}"
    )

    print(
        f"Actual rows   : {len(feature_df)}"
    )

    if len(feature_df) == expected_rows:
        print("Row count check: PASS")
    else:
        print("Row count check: FAIL")

    if target_missing == 0:
        print("Utility target check: PASS")
    else:
        print("Utility target check: FAIL")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()