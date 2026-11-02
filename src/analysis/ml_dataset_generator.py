"""
ML Dataset Generator

Combines:
1. Cover-image features
2. Payload features
3. Traditional steganography experiment results

into a single dataset for training the ML method recommender.
"""

from pathlib import Path

import pandas as pd

from src.analysis.image_features import extract_image_features
from src.analysis.payload_features import extract_payload_features


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_RESULTS_PATH = (
    PROJECT_ROOT / "results" / "experimental_results.csv"
)

OUTPUT_DATASET_PATH = (
    PROJECT_ROOT / "results" / "ml_dataset.csv"
)

COVER_IMAGE_PATH = (
    PROJECT_ROOT / "data" / "images" / "processed" / "cover_processed.png"
)

PAYLOAD_DIRECTORY = (
    PROJECT_ROOT / "data" / "payloads"
)


# ============================================================
# FIND PAYLOAD FILE
# ============================================================

def find_payload_file(payload_type, payload_extension):
    """
    Find the payload file corresponding to an experiment.

    Example:
        TEXT + txt  -> test.txt
        AUDIO + wav -> test.wav
        IMAGE + png -> test.png
    """

    payload_extension = str(payload_extension).lower().strip()

    # First try the common test payload naming convention.
    test_file = PAYLOAD_DIRECTORY / f"test.{payload_extension}"

    if test_file.exists():
        return test_file

    # Otherwise search for any matching extension.
    matching_files = list(
        PAYLOAD_DIRECTORY.glob(f"*.{payload_extension}")
    )

    if matching_files:
        return matching_files[0]

    raise FileNotFoundError(
        f"No payload file found for "
        f"type={payload_type}, extension={payload_extension}"
    )


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def generate_ml_dataset():

    print("=" * 70)
    print("ML DATASET GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------------

    if not EXPERIMENT_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Experiment results not found:\n"
            f"{EXPERIMENT_RESULTS_PATH}"
        )

    if not COVER_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Cover image not found:\n"
            f"{COVER_IMAGE_PATH}"
        )

    if not PAYLOAD_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Payload directory not found:\n"
            f"{PAYLOAD_DIRECTORY}"
        )

    # --------------------------------------------------------
    # LOAD EXPERIMENT RESULTS
    # --------------------------------------------------------

    experiments_df = pd.read_csv(EXPERIMENT_RESULTS_PATH)

    print(
        f"\nExperiments loaded : {len(experiments_df)}"
    )

    # --------------------------------------------------------
    # DISPLAY AVAILABLE COLUMNS
    # --------------------------------------------------------

    print("\nAvailable experiment columns:")

    for column in experiments_df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # EXTRACT COVER IMAGE FEATURES
    # --------------------------------------------------------

    print("\nExtracting cover-image features...")

    cover_features = extract_image_features(
        COVER_IMAGE_PATH
    )

    print("✓ Cover features extracted")

    # --------------------------------------------------------
    # PROCESS EACH EXPERIMENT
    # --------------------------------------------------------

    dataset_rows = []

    for index, row in experiments_df.iterrows():

        print(
            f"\nProcessing experiment "
            f"{index + 1}/{len(experiments_df)}"
        )

        try:

            # ==================================================
            # BASIC EXPERIMENT INFORMATION
            # ==================================================

            method = row["method"]
            payload_type = row["payload_type"]

            # IMPORTANT:
            # The actual CSV column is payload_extension.
            payload_extension = row["payload_extension"]

            # ==================================================
            # FIND PAYLOAD FILE
            # ==================================================

            payload_path = find_payload_file(
                payload_type,
                payload_extension
            )

            print(
                f"  Payload : {payload_path.name}"
            )

            # ==================================================
            # EXTRACT PAYLOAD FEATURES
            # ==================================================

            payload_features = extract_payload_features(
                payload_path
            )

            print(
                f"  Payload type : "
                f"{payload_features['payload_type']}"
            )

            # ==================================================
            # CREATE COMBINED ROW
            # ==================================================

            combined_row = {}

            # --------------------------------------------------
            # COVER IMAGE FEATURES
            # --------------------------------------------------

            for key, value in cover_features.items():
                combined_row[key] = value

            # --------------------------------------------------
            # PAYLOAD FEATURES
            # --------------------------------------------------

            for key, value in payload_features.items():

                # Avoid duplicating payload_type because
                # experiment results already contain it.
                if key != "payload_type":
                    combined_row[key] = value

            # --------------------------------------------------
            # EXPERIMENT FEATURES
            # --------------------------------------------------

            for column in experiments_df.columns:

                combined_row[column] = row[column]

            # --------------------------------------------------
            # ADD METHOD LABEL
            # --------------------------------------------------

            combined_row["recommended_method"] = method

            # --------------------------------------------------
            # ADD ROW
            # --------------------------------------------------

            dataset_rows.append(combined_row)

            print("  ✓ Experiment added")

        except Exception as error:

            print(
                f"\n  ✗ ERROR: {error}"
            )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not dataset_rows:

        raise RuntimeError(
            "No dataset rows were generated."
        )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    dataset_df = pd.DataFrame(
        dataset_rows
    )

    # ========================================================
    # REMOVE DUPLICATE COLUMNS
    # ========================================================

    dataset_df = dataset_df.loc[
        :,
        ~dataset_df.columns.duplicated()
    ]

    # ========================================================
    # SAVE DATASET
    # ========================================================

    OUTPUT_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset_df.to_csv(
        OUTPUT_DATASET_PATH,
        index=False
    )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ML DATASET GENERATED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nRows    : {dataset_df.shape[0]}"
    )

    print(
        f"Columns : {dataset_df.shape[1]}"
    )

    print(
        f"\nOutput file:\n"
        f"{OUTPUT_DATASET_PATH}"
    )

    print("\nDataset columns:")

    for column in dataset_df.columns:
        print(f"  - {column}")

    print("\nDataset preview:")
    print(
        dataset_df.head().to_string()
    )

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_ml_dataset()