from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

DATASET_FILE = Path(
    "results/multi_cover_analysis/recommender_feature_dataset.csv"
)

OUTPUT_DIR = Path(
    "results/multi_cover_analysis/recommender_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "method",

    # Cover
    "cover_cover_width",
    "cover_cover_height",
    "cover_cover_pixels",
    "cover_mean_intensity",
    "cover_std_intensity",
    "cover_entropy",
    "cover_edge_density",

    # Payload
    "payload_type",
    "payload_payload_size_bytes",
    "payload_text_length",
    "payload_duration_seconds",
    "payload_sample_rate",
    "payload_channels",
    "payload_sample_width_bytes",
    "payload_audio_frames",
    "payload_payload_width",
    "payload_payload_height",
    "payload_payload_pixels",

    # Capacity
    "capacity_bytes",
    "required_bits",
    "capacity_fit_numeric",
    "capacity_utilization_percent",
]

TARGET = "utility_score"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATASET_FILE)

    print(f"Dataset shape: {df.shape}")

    covers = sorted(
        df["cover_name"].unique()
    )

    train_covers = covers[:8]
    test_covers = covers[8:]

    train_df = df[
        df["cover_name"].isin(train_covers)
    ].copy()

    test_df = df[
        df["cover_name"].isin(test_covers)
    ].copy()

    return df, train_df, test_df


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(features):

    categorical = [
        "method",
        "payload_type",
    ]

    numeric = [
        f for f in features
        if f not in categorical
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# ============================================================
# REGRESSION DIAGNOSTICS
# ============================================================

def regression_diagnostics(
    model,
    test_df
):

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    print("\n" + "=" * 70)
    print("REGRESSION DIAGNOSTICS")
    print("=" * 70)

    print(f"\nMAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")

    # --------------------------------------------------------
    # Actual vs predicted plot
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Utility")
    plt.ylabel("Predicted Utility")
    plt.title("Actual vs Predicted Utility")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_df = test_df[
        [
            "cover_name",
            "payload_name",
            "payload_type",
            "method",
            "utility_score",
            "capacity_fit_numeric",
        ]
    ].copy()

    prediction_df[
        "predicted_utility"
    ] = predictions

    prediction_df[
        "absolute_error"
    ] = (
        prediction_df["utility_score"]
        - prediction_df["predicted_utility"]
    ).abs()

    prediction_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    return prediction_df


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def feature_importance(model):

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    rf = model.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = rf.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        importance_df.head(20)
        .to_string(index=False)
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot top 15
    # --------------------------------------------------------

    top = importance_df.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top["feature"][::-1],
        top["importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=200
    )

    plt.close()

    return importance_df


# ============================================================
# PER-METHOD ERROR
# ============================================================

def method_error_analysis(
    prediction_df
):

    print("\n" + "=" * 70)
    print("ERROR BY METHOD")
    print("=" * 70)

    prediction_df["error"] = (
        prediction_df["utility_score"]
        - prediction_df["predicted_utility"]
    ).abs()

    summary = (
        prediction_df
        .groupby("method")["error"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "max"
            ]
        )
        .round(6)
    )

    print(summary)

    summary.to_csv(
        OUTPUT_DIR / "method_error_summary.csv"
    )


# ============================================================
# ALL THREE CANDIDATE PREDICTIONS
# ============================================================

def candidate_predictions(
    model,
    test_df
):

    print("\n" + "=" * 70)
    print("CANDIDATE METHOD PREDICTIONS")
    print("=" * 70)

    rows = []

    scenarios = (
        test_df[
            [
                "cover_name",
                "payload_name",
                "payload_type"
            ]
        ]
        .drop_duplicates()
    )

    for _, scenario in scenarios.iterrows():

        cover = scenario["cover_name"]
        payload = scenario["payload_name"]

        scenario_df = test_df[
            (test_df["cover_name"] == cover)
            &
            (test_df["payload_name"] == payload)
        ].copy()

        X = scenario_df[FEATURES]

        predictions = model.predict(X)

        for i, (_, row) in enumerate(
            scenario_df.iterrows()
        ):

            rows.append({
                "cover_name": cover,
                "payload_name": payload,
                "payload_type":
                    row["payload_type"],
                "method":
                    row["method"],
                "actual_utility":
                    row["utility_score"],
                "predicted_utility":
                    predictions[i],
                "capacity_fit":
                    row["capacity_fit_numeric"],
            })

    result = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Add predicted winner
    # --------------------------------------------------------

    winners = []

    for (
        cover,
        payload
    ), group in result.groupby(
        [
            "cover_name",
            "payload_name"
        ]
    ):

        winner = group.loc[
            group["predicted_utility"].idxmax()
        ]

        actual = group.loc[
            group["actual_utility"].idxmax()
        ]

        winners.append({
            "cover_name": cover,
            "payload_name": payload,
            "actual_best_method":
                actual["method"],
            "predicted_best_method":
                winner["method"],
            "actual_best_utility":
                actual["actual_utility"],
            "predicted_best_utility":
                winner["predicted_utility"],
            "correct":
                actual["method"]
                ==
                winner["method"]
        })

    winner_df = pd.DataFrame(winners)

    print("\nPredicted winners:")

    print(
        winner_df[
            [
                "cover_name",
                "payload_name",
                "actual_best_method",
                "predicted_best_method",
                "correct"
            ]
        ]
        .to_string(index=False)
    )

    accuracy = (
        winner_df["correct"].mean()
        * 100
    )

    print(
        f"\nRecommendation accuracy: "
        f"{accuracy:.2f}%"
    )

    result.to_csv(
        OUTPUT_DIR / "candidate_predictions.csv",
        index=False
    )

    winner_df.to_csv(
        OUTPUT_DIR / "recommendation_results.csv",
        index=False
    )

    return result


# ============================================================
# ABLATION TEST
# ============================================================

def ablation_test(
    train_df,
    test_df
):

    print("\n" + "=" * 70)
    print("ABLATION TEST — WITHOUT CAPACITY FEATURES")
    print("=" * 70)

    reduced_features = [
        feature
        for feature in FEATURES
        if feature not in [
            "capacity_bytes",
            "required_bits",
            "capacity_fit_numeric",
            "capacity_utilization_percent",
        ]
    ]

    model = build_model(
        reduced_features
    )

    model.fit(
        train_df[reduced_features],
        train_df[TARGET]
    )

    predictions = model.predict(
        test_df[reduced_features]
    )

    mae = mean_absolute_error(
        test_df[TARGET],
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test_df[TARGET],
            predictions
        )
    )

    print(
        f"\nWithout capacity features:"
    )

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    # --------------------------------------------------------
    # Recommendation accuracy
    # --------------------------------------------------------

    temp = test_df[
        [
            "cover_name",
            "payload_name",
            "method",
            "utility_score"
        ]
    ].copy()

    temp["predicted"] = predictions

    correct = 0
    total = 0

    for (
        cover,
        payload
    ), group in temp.groupby(
        [
            "cover_name",
            "payload_name"
        ]
    ):

        actual_best = group.loc[
            group["utility_score"].idxmax()
        ]["method"]

        predicted_best = group.loc[
            group["predicted"].idxmax()
        ]["method"]

        if actual_best == predicted_best:
            correct += 1

        total += 1

    accuracy = (
        correct / total * 100
    )

    print(
        f"Recommendation accuracy "
        f"without capacity: {accuracy:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RECOMMENDER MODEL DIAGNOSTICS")
    print("=" * 70)

    df, train_df, test_df = load_data()

    print(
        f"\nTraining rows: {len(train_df)}"
    )

    print(
        f"Testing rows : {len(test_df)}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = build_model(FEATURES)

    model.fit(
        train_df[FEATURES],
        train_df[TARGET]
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    prediction_df = regression_diagnostics(
        model,
        test_df
    )

    feature_importance(model)

    method_error_analysis(
        prediction_df
    )

    candidate_predictions(
        model,
        test_df
    )

    ablation_test(
        train_df,
        test_df
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()