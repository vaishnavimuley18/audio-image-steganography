from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = Path(
    "results/multi_cover_analysis/recommender_model.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_recommender():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Recommender model not found:\n{MODEL_FILE}"
        )

    saved_data = joblib.load(MODEL_FILE)

    model = saved_data["model"]
    features = saved_data["features"]

    return model, features


# ============================================================
# PREDICT UTILITY
# ============================================================

def predict_method_utilities(candidate_rows):

    """
    Predict utility score for LSB, DCT and DWT.

    candidate_rows:
        List of dictionaries.
        Each dictionary represents one candidate method.

    Returns:
        DataFrame containing predicted utilities.
    """

    model, features = load_recommender()

    df = pd.DataFrame(candidate_rows)

    # Make sure the exact training feature order is used.
    df = df[features]

    predictions = model.predict(df)

    result = pd.DataFrame({
        "method": df["method"].values,
        "predicted_utility": predictions
    })

    return result


# ============================================================
# SELECT BEST METHOD
# ============================================================

def recommend_method(candidate_rows):

    """
    Select the best feasible method.

    Methods that cannot fit the payload are removed before
    selecting the final recommendation.
    """

    predictions = predict_method_utilities(
        candidate_rows
    )

    # --------------------------------------------------------
    # Add capacity information
    # --------------------------------------------------------

    original_df = pd.DataFrame(candidate_rows)

    predictions["capacity_fit_numeric"] = (
        original_df["capacity_fit_numeric"].values
    )

    predictions["capacity_utilization_percent"] = (
        original_df["capacity_utilization_percent"].values
    )

    # --------------------------------------------------------
    # Remove methods that cannot fit
    # --------------------------------------------------------

    feasible = predictions[
        predictions["capacity_fit_numeric"] == 1
    ].copy()

    # --------------------------------------------------------
    # No method available
    # --------------------------------------------------------

    if feasible.empty:

        return {
            "recommended_method": None,
            "reason": "Payload exceeds the capacity of all methods.",
            "candidates": predictions
        }

    # --------------------------------------------------------
    # Highest predicted utility
    # --------------------------------------------------------

    best_index = feasible[
        "predicted_utility"
    ].idxmax()

    best_method = feasible.loc[
        best_index,
        "method"
    ]

    best_utility = feasible.loc[
        best_index,
        "predicted_utility"
    ]

    return {
        "recommended_method": best_method,
        "predicted_utility": float(best_utility),
        "reason": "Highest predicted utility among feasible methods.",
        "candidates": predictions
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RECOMMENDER MODEL TEST")
    print("=" * 70)

    model, features = load_recommender()

    print("\nModel loaded successfully.")
    print(f"Number of features: {len(features)}")

    print("\nFeatures used:")

    for i, feature in enumerate(features, start=1):
        print(f"{i:2}. {feature}")

    print("\nModel is ready for inference.")