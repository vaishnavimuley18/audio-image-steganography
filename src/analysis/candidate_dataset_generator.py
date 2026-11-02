from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "multi_cover_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "multi_cover_analysis"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "candidate_ml_dataset.csv"
)


# ============================================================
# UTILITY WEIGHTS
# ============================================================

WEIGHTS = {
    "reliability": 0.30,
    "psnr": 0.25,
    "ssim": 0.20,
    "embedding_speed": 0.10,
    "extraction_speed": 0.05,
    "distortion": 0.05,
    "capacity_margin": 0.05,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def min_max_score(series):
    """
    Normalize a series to [0, 1].

    Higher value = better.
    """
    series = pd.to_numeric(series, errors="coerce")

    valid = series.dropna()

    if len(valid) == 0:
        return pd.Series(0.0, index=series.index)

    minimum = valid.min()
    maximum = valid.max()

    if maximum == minimum:
        return pd.Series(1.0, index=series.index)

    result = (series - minimum) / (maximum - minimum)

    return result.fillna(0.0)


def inverse_min_max_score(series):
    """
    Normalize a series to [0, 1].

    Lower value = better.
    """
    series = pd.to_numeric(series, errors="coerce")

    valid = series.dropna()

    if len(valid) == 0:
        return pd.Series(0.0, index=series.index)

    minimum = valid.min()
    maximum = valid.max()

    if maximum == minimum:
        return pd.Series(1.0, index=series.index)

    result = (maximum - series) / (maximum - minimum)

    return result.fillna(0.0)


# ============================================================
# LOAD DATA
# ============================================================

def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("CANDIDATE DATASET GENERATOR")
    print("=" * 70)

    print(f"Input file : {INPUT_FILE}")
    print(f"Dataset shape : {df.shape}")

    return df


# ============================================================
# PREPARE OUTCOME FEATURES
# ============================================================

def prepare_outcome_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Basic status indicators
    # --------------------------------------------------------

    df["is_success"] = (
        df["status"].astype(str).str.upper() == "SUCCESS"
    ).astype(int)

    df["capacity_fit_numeric"] = (
        df["capacity_fit"]
        .astype(str)
        .str.lower()
        .map({
            "true": 1,
            "false": 0
        })
        .fillna(0)
    )

    # --------------------------------------------------------
    # Reliability
    #
    # Successful extraction with 100% integrity gets 1.
    # Partial integrity gets proportional reliability.
    # Capacity failures get 0.
    # --------------------------------------------------------

    integrity = pd.to_numeric(
        df["integrity_percent"],
        errors="coerce"
    ).fillna(0)

    df["reliability_score"] = (
        integrity / 100.0
    )

    # Capacity failure means the method cannot be used.
    df.loc[
        df["capacity_fit_numeric"] == 0,
        "reliability_score"
    ] = 0.0

    # --------------------------------------------------------
    # Numeric metrics
    # --------------------------------------------------------

    df["psnr_numeric"] = pd.to_numeric(
        df["psnr"],
        errors="coerce"
    )

    df["ssim_numeric"] = pd.to_numeric(
        df["ssim"],
        errors="coerce"
    )

    df["mse_numeric"] = pd.to_numeric(
        df["mse"],
        errors="coerce"
    )

    df["embedding_time_numeric"] = pd.to_numeric(
        df["embedding_time_seconds"],
        errors="coerce"
    )

    df["extraction_time_numeric"] = pd.to_numeric(
        df["extraction_time_seconds"],
        errors="coerce"
    )

    df["capacity_utilization_numeric"] = pd.to_numeric(
        df["capacity_utilization_percent"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Capacity margin
    #
    # Lower utilization means more remaining capacity.
    #
    # Example:
    # 10% utilization -> high margin
    # 90% utilization -> low margin
    # --------------------------------------------------------

    utilization = df["capacity_utilization_numeric"]

    df["capacity_margin_raw"] = (
        1.0 - (utilization / 100.0)
    )

    df["capacity_margin_raw"] = (
        df["capacity_margin_raw"]
        .clip(lower=0.0, upper=1.0)
        .fillna(0.0)
    )

    return df


# ============================================================
# CALCULATE GLOBAL NORMALIZED SCORES
# ============================================================

def calculate_normalized_scores(df):

    df = df.copy()

    # --------------------------------------------------------
    # PSNR
    # Higher is better
    # --------------------------------------------------------

    df["psnr_score"] = min_max_score(
        df["psnr_numeric"]
    )

    # --------------------------------------------------------
    # SSIM
    # Higher is better
    # --------------------------------------------------------

    df["ssim_score"] = min_max_score(
        df["ssim_numeric"]
    )

    # --------------------------------------------------------
    # Embedding time
    # Lower is better
    # --------------------------------------------------------

    df["embedding_speed_score"] = inverse_min_max_score(
        df["embedding_time_numeric"]
    )

    # --------------------------------------------------------
    # Extraction time
    # Lower is better
    # --------------------------------------------------------

    df["extraction_speed_score"] = inverse_min_max_score(
        df["extraction_time_numeric"]
    )

    # --------------------------------------------------------
    # Distortion
    #
    # MSE is used as distortion.
    # Lower is better.
    # --------------------------------------------------------

    df["distortion_score"] = inverse_min_max_score(
        df["mse_numeric"]
    )

    # --------------------------------------------------------
    # Capacity margin
    #
    # Higher remaining capacity is better.
    # --------------------------------------------------------

    df["capacity_margin_score"] = (
        df["capacity_margin_raw"]
    )

    return df


# ============================================================
# CALCULATE UTILITY
# ============================================================

def calculate_utility(df):

    df = df.copy()

    # --------------------------------------------------------
    # Weighted utility
    # --------------------------------------------------------

    df["utility_score"] = (
        WEIGHTS["reliability"]
        * df["reliability_score"]

        + WEIGHTS["psnr"]
        * df["psnr_score"]

        + WEIGHTS["ssim"]
        * df["ssim_score"]

        + WEIGHTS["embedding_speed"]
        * df["embedding_speed_score"]

        + WEIGHTS["extraction_speed"]
        * df["extraction_speed_score"]

        + WEIGHTS["distortion"]
        * df["distortion_score"]

        + WEIGHTS["capacity_margin"]
        * df["capacity_margin_score"]
    )

    # --------------------------------------------------------
    # A capacity failure is not a usable candidate.
    # --------------------------------------------------------

    df.loc[
        df["capacity_fit_numeric"] == 0,
        "utility_score"
    ] = 0.0

    # --------------------------------------------------------
    # Keep utility within [0, 1]
    # --------------------------------------------------------

    df["utility_score"] = (
        df["utility_score"]
        .clip(lower=0.0, upper=1.0)
    )

    return df


# ============================================================
# CREATE SCENARIO RANKING
# ============================================================

def create_rankings(df):

    df = df.copy()

    # --------------------------------------------------------
    # Rank methods inside each
    # cover + payload scenario
    # --------------------------------------------------------

    df["utility_rank"] = (
        df.groupby(
            ["cover_name", "payload_name"]
        )["utility_score"]
        .rank(
            method="min",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Best method for each scenario
    # --------------------------------------------------------

    df["is_best_candidate"] = (
        df["utility_rank"] == 1
    ).astype(int)

    return df


# ============================================================
# CREATE SCENARIO TARGET TABLE
# ============================================================

def create_recommendation_targets(df):

    rows = []

    grouped = df.groupby(
        ["cover_name", "payload_name"],
        sort=True
    )

    for (cover_name, payload_name), group in grouped:

        group = group.sort_values(
            "utility_score",
            ascending=False
        )

        best = group.iloc[0]

        rows.append({
            "cover_name": cover_name,
            "payload_name": payload_name,
            "payload_type": best["payload_type"],
            "recommended_method": best["method"],
            "best_utility_score": best["utility_score"],
            "candidate_count": len(group),
            "successful_candidates": int(
                (group["utility_score"] > 0).sum()
            )
        })

    targets = pd.DataFrame(rows)

    return targets


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df, targets):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Candidate-level dataset
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Scenario-level recommendation targets
    # --------------------------------------------------------

    target_file = (
        OUTPUT_DIR
        / "utility_recommendation_targets.csv"
    )

    targets.to_csv(
        target_file,
        index=False
    )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(f"Candidate dataset:")
    print(OUTPUT_FILE)

    print()
    print("Recommendation targets:")
    print(target_file)

    return target_file


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(df, targets):

    print()
    print("=" * 70)
    print("UTILITY SCORE SUMMARY")
    print("=" * 70)

    print(
        df[
            [
                "method",
                "utility_score"
            ]
        ]
        .groupby("method")
        .agg(
            count=("utility_score", "count"),
            mean=("utility_score", "mean"),
            max=("utility_score", "max"),
            min=("utility_score", "min")
        )
        .round(4)
    )

    print()
    print("=" * 70)
    print("BEST METHOD DISTRIBUTION")
    print("=" * 70)

    print(
        targets["recommended_method"]
        .value_counts()
    )

    print()
    print("=" * 70)
    print("UTILITY SCORE DISTRIBUTION")
    print("=" * 70)

    print(
        df["utility_score"]
        .describe()
        .round(4)
    )

    print()
    print("=" * 70)
    print("EXAMPLE RECOMMENDATIONS")
    print("=" * 70)

    print(
        targets[
            [
                "cover_name",
                "payload_name",
                "payload_type",
                "recommended_method",
                "best_utility_score",
                "successful_candidates"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_results()

    df = prepare_outcome_features(df)

    df = calculate_normalized_scores(df)

    df = calculate_utility(df)

    df = create_rankings(df)

    targets = create_recommendation_targets(df)

    save_dataset(
        df,
        targets
    )

    display_summary(
        df,
        targets
    )


if __name__ == "__main__":
    main()