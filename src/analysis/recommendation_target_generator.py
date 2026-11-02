from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "results" / "multi_cover_results.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "multi_cover_analysis"
OUTPUT_FILE = OUTPUT_DIR / "recommendation_targets.csv"


# ============================================================
# SETTINGS
# ============================================================

METHODS = ["LSB", "DCT", "DWT"]

# Weighting used for the utility score
WEIGHTS = {
    "reliability": 0.40,
    "psnr": 0.20,
    "ssim": 0.15,
    "embedding_speed": 0.10,
    "extraction_speed": 0.05,
    "distortion": 0.05,
    "capacity_margin": 0.05,
}


# ============================================================
# NORMALIZATION
# ============================================================

def min_max_normalize(series):
    """
    Normalize values to 0-1.

    Higher value = better.
    """
    series = series.astype(float)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (series - min_value) / (max_value - min_value)


def inverse_min_max_normalize(series):
    """
    Normalize values to 0-1 where LOWER is better.

    Used for execution time and distortion.
    """
    normalized = min_max_normalize(series)
    return 1 - normalized


# ============================================================
# LOAD DATA
# ============================================================

def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 65)
    print("RECOMMENDATION TARGET GENERATOR")
    print("=" * 65)

    print(f"Input file : {INPUT_FILE}")
    print(f"Dataset    : {df.shape}")

    return df


# ============================================================
# DETERMINE ELIGIBILITY
# ============================================================

def determine_eligibility(df):
    """
    A method is eligible when:

    1. Capacity is sufficient
    2. Experiment succeeded
    3. Integrity is 100%
    """

    df = df.copy()

    df["eligible"] = (
        (df["status"] == "SUCCESS")
        & (df["capacity_fit"] == True)
        & (df["integrity_percent"] >= 99.999)
    )

    return df


# ============================================================
# CALCULATE UTILITY SCORE
# ============================================================

def calculate_utility_scores(group):
    """
    Calculate utility score for all methods within
    one cover + payload scenario.

    Only eligible methods receive a meaningful score.
    """

    group = group.copy()

    eligible = group["eligible"]

    # Default score for failed/ineligible methods
    group["utility_score"] = 0.0

    if eligible.sum() == 0:
        return group

    valid = group.loc[eligible].copy()

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    valid["reliability_score"] = 1.0

    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    valid["psnr_score"] = min_max_normalize(valid["psnr"])

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    valid["ssim_score"] = min_max_normalize(valid["ssim"])

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    valid["embedding_speed_score"] = inverse_min_max_normalize(
        valid["embedding_time_seconds"]
    )

    valid["extraction_speed_score"] = inverse_min_max_normalize(
        valid["extraction_time_seconds"]
    )

    # --------------------------------------------------------
    # Distortion
    # --------------------------------------------------------

    valid["distortion_score"] = inverse_min_max_normalize(
        valid["mse"]
    )

    # --------------------------------------------------------
    # Capacity margin
    #
    # More unused capacity = better.
    # capacity_fit is already guaranteed here.
    # --------------------------------------------------------

    valid["capacity_margin_percent"] = (
        100 - valid["capacity_utilization_percent"]
    )

    valid["capacity_margin_score"] = min_max_normalize(
        valid["capacity_margin_percent"]
    )

    # --------------------------------------------------------
    # Final utility score
    # --------------------------------------------------------

    valid["utility_score"] = (
        WEIGHTS["reliability"]
        * valid["reliability_score"]

        + WEIGHTS["psnr"]
        * valid["psnr_score"]

        + WEIGHTS["ssim"]
        * valid["ssim_score"]

        + WEIGHTS["embedding_speed"]
        * valid["embedding_speed_score"]

        + WEIGHTS["extraction_speed"]
        * valid["extraction_speed_score"]

        + WEIGHTS["distortion"]
        * valid["distortion_score"]

        + WEIGHTS["capacity_margin"]
        * valid["capacity_margin_score"]
    )

    # Put calculated values back into original group
    for column in [
        "reliability_score",
        "psnr_score",
        "ssim_score",
        "embedding_speed_score",
        "extraction_speed_score",
        "distortion_score",
        "capacity_margin_percent",
        "capacity_margin_score",
        "utility_score",
    ]:
        group.loc[valid.index, column] = valid[column]

    return group


# ============================================================
# GENERATE TARGET
# ============================================================

def generate_targets(df):
    """
    Generate recommendation targets for every
    cover + payload scenario.

    Each scenario contains up to three candidate methods:
    LSB, DCT and DWT.

    A method is eligible only if:
        - capacity is sufficient
        - experiment succeeded
        - integrity is 100%
    """

    df = determine_eligibility(df)

    # --------------------------------------------------------
    # Calculate utility scores separately for each
    # cover + payload scenario.
    #
    # We explicitly concatenate the groups afterward.
    # This avoids pandas groupby/apply removing the grouping
    # columns.
    # --------------------------------------------------------

    scored_groups = []

    for (cover_name, payload_name), group in df.groupby(
        ["cover_name", "payload_name"],
        sort=True
    ):

        scored_group = calculate_utility_scores(
            group.copy()
        )

        scored_groups.append(scored_group)

    if not scored_groups:
        raise ValueError(
            "No cover + payload groups were found."
        )

    df = pd.concat(
        scored_groups,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Select the best method for every scenario
    # --------------------------------------------------------

    target_rows = []

    for (cover_name, payload_name), group in df.groupby(
        ["cover_name", "payload_name"],
        sort=True
    ):

        eligible = group[
            group["eligible"] == True
        ]

        # ----------------------------------------------------
        # No eligible method
        # ----------------------------------------------------

        if len(eligible) == 0:

            recommended_method = "NONE"
            best_score = 0.0
            eligible_methods = ""

        # ----------------------------------------------------
        # Select highest utility score
        # ----------------------------------------------------

        else:

            best_index = eligible[
                "utility_score"
            ].idxmax()

            best_row = eligible.loc[
                best_index
            ]

            recommended_method = best_row[
                "method"
            ]

            best_score = best_row[
                "utility_score"
            ]

            eligible_methods = ",".join(
                eligible["method"].tolist()
            )

        # ----------------------------------------------------
        # Create one recommendation-target row
        # ----------------------------------------------------

        target_rows.append({

            "cover_name":
                cover_name,

            "payload_name":
                payload_name,

            "payload_type":
                group["payload_type"].iloc[0],

            "recommended_method":
                recommended_method,

            "best_utility_score":
                best_score,

            "eligible_methods":
                eligible_methods,
        })

    targets = pd.DataFrame(
        target_rows
    )

    return df, targets
# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_results()

    analyzed_df, targets = generate_targets(df)

    # --------------------------------------------------------
    # Save candidate-level scores
    # --------------------------------------------------------

    candidate_file = (
        OUTPUT_DIR / "recommendation_candidate_scores.csv"
    )

    analyzed_df.to_csv(
        candidate_file,
        index=False
    )

    # --------------------------------------------------------
    # Save recommendation targets
    # --------------------------------------------------------

    targets.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("RECOMMENDATION TARGETS")
    print("=" * 65)

    print(
        targets[
            [
                "cover_name",
                "payload_name",
                "payload_type",
                "recommended_method",
                "best_utility_score",
                "eligible_methods",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("RECOMMENDATION DISTRIBUTION")
    print("=" * 65)

    print(
        targets["recommended_method"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 65)
    print("FILES CREATED")
    print("=" * 65)

    print(candidate_file)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()