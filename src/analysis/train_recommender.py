from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

DATASET_FILE = Path(
    "results/multi_cover_analysis/recommender_feature_dataset.csv"
)

MODEL_FILE = Path(
    "results/multi_cover_analysis/recommender_model.joblib"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    # Candidate method
    "method",

    # Cover features
    "cover_cover_width",
    "cover_cover_height",
    "cover_cover_pixels",
    "cover_mean_intensity",
    "cover_std_intensity",
    "cover_entropy",
    "cover_edge_density",

    # Payload features
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

    # Capacity features
    "capacity_bytes",
    "required_bits",
    "capacity_fit_numeric",
    "capacity_utilization_percent",
]

TARGET = "utility_score"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    df = pd.read_csv(DATASET_FILE)

    print(f"Dataset loaded: {df.shape}")

    return df


# ============================================================
# GROUPED SPLIT
# ============================================================

def grouped_split(df):

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

    print("\nTRAINING COVERS")

    for cover in train_covers:
        print(f"  {cover}")

    print("\nTESTING COVERS")

    for cover in test_covers:
        print(f"  {cover}")

    print(f"\nTraining rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    return train_df, test_df


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor():

    categorical_features = [
        "method",
        "payload_type",
    ]

    numeric_features = [
        feature
        for feature in FEATURES
        if feature not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
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
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    preprocessor = build_preprocessor()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

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

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"\nMAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")

    return predictions


# ============================================================
# CANDIDATE RANKING
# ============================================================

def evaluate_recommendations(
    test_df,
    predictions
):

    results = test_df[
        [
            "cover_name",
            "payload_name",
            "payload_type",
            "method",
            "utility_score",
            "capacity_fit_numeric",
        ]
    ].copy()

    results["predicted_utility"] = predictions

    # --------------------------------------------------------
    # Actual best method
    # --------------------------------------------------------

    actual_best = (
        results.loc[
            results.groupby(
                [
                    "cover_name",
                    "payload_name"
                ]
            )["utility_score"].idxmax()
        ]
        [
            [
                "cover_name",
                "payload_name",
                "method",
                "utility_score",
            ]
        ]
        .rename(
            columns={
                "method": "actual_best_method",
                "utility_score": "actual_best_utility",
            }
        )
    )

    # --------------------------------------------------------
    # Predicted best method
    # --------------------------------------------------------

    predicted_best = (
        results.loc[
            results.groupby(
                [
                    "cover_name",
                    "payload_name"
                ]
            )["predicted_utility"].idxmax()
        ]
        [
            [
                "cover_name",
                "payload_name",
                "method",
                "predicted_utility",
            ]
        ]
        .rename(
            columns={
                "method": "predicted_best_method",
                "predicted_utility":
                    "predicted_best_utility",
            }
        )
    )

    comparison = actual_best.merge(
        predicted_best,
        on=[
            "cover_name",
            "payload_name"
        ]
    )

    comparison["recommendation_correct"] = (
        comparison["actual_best_method"]
        ==
        comparison["predicted_best_method"]
    )

    print("\n" + "=" * 70)
    print("RECOMMENDATION RESULTS")
    print("=" * 70)

    print(
        "\nCorrect recommendations:"
        f" {comparison['recommendation_correct'].sum()}"
        f" / {len(comparison)}"
    )

    accuracy = (
        comparison["recommendation_correct"]
        .mean()
        * 100
    )

    print(
        f"Recommendation accuracy: {accuracy:.2f}%"
    )

    print("\nDetailed recommendations:")

    print(
        comparison.to_string(
            index=False
        )
    )

    return comparison


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "model_type": "RandomForestRegressor",
            "version": 1
        },
        MODEL_FILE
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(f"\nFile: {MODEL_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ML RECOMMENDER — RANDOM FOREST REGRESSOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Split by cover
    # --------------------------------------------------------

    train_df, test_df = grouped_split(df)

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\nFeature count:", len(FEATURES))

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Evaluate regression
    # --------------------------------------------------------

    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Evaluate recommendation
    # --------------------------------------------------------

    evaluate_recommendations(
        test_df,
        predictions
    )

    # --------------------------------------------------------
    # Save fitted model
    # --------------------------------------------------------

    save_model(model)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING + MODEL EXPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()