from pathlib import Path
import math
import wave

import cv2
import numpy as np
from PIL import Image

from src.traditional import lsb, dct, dwt
from src.analysis.recommender_model import (
    predict_method_utilities,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMP_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "processed"
    / "auto_recommend"
)


# ============================================================
# METHODS
# ============================================================

METHODS = {
    "LSB": lsb,
    "DCT": dct,
    "DWT": dwt,
}


# ============================================================
# PAYLOAD TYPE
# ============================================================

def detect_payload_type(payload_path):
    """
    Detect payload type from file extension.
    """

    extension = payload_path.suffix.lower()

    if extension == ".txt":
        return "TEXT"

    if extension == ".wav":
        return "AUDIO"

    if extension in [".png", ".jpg", ".jpeg"]:
        return "IMAGE"

    raise ValueError(
        f"Unsupported payload type: {extension}"
    )


# ============================================================
# COVER PREPROCESSING
# ============================================================

def preprocess_cover(cover_path):

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        TEMP_DIR
        / f"{cover_path.stem}_processed.png"
    )

    with Image.open(cover_path) as image:

        image = image.convert("RGB")

        target_size = 4000

        # Preserve aspect ratio
        scale = min(
            target_size / image.width,
            target_size / image.height
        )

        new_width = int(
            image.width * scale
        )

        new_height = int(
            image.height * scale
        )

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        # Create 4000x4000 RGB canvas
        canvas = Image.new(
            "RGB",
            (target_size, target_size),
            (0, 0, 0)
        )

        x = (
            target_size - new_width
        ) // 2

        y = (
            target_size - new_height
        ) // 2

        canvas.paste(
            resized,
            (x, y)
        )

        canvas.save(
            output_path,
            format="PNG"
        )

    return output_path


# ============================================================
# COVER FEATURES
# ============================================================

def calculate_entropy(image):

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.ravel()

    total = histogram.sum()

    if total == 0:
        return 0.0

    probabilities = (
        histogram / total
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return float(entropy)


def calculate_edge_density(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_pixels = np.count_nonzero(edges)

    total_pixels = edges.size

    if total_pixels == 0:
        return 0.0

    return float(
        edge_pixels / total_pixels
    )


def extract_cover_features(cover_path):

    image = cv2.imread(
        str(cover_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load cover image:\n{cover_path}"
        )

    height, width, channels = image.shape

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return {
        "cover_cover_width": width,
        "cover_cover_height": height,
        "cover_cover_pixels": width * height,

        # Keep constant feature from training dataset.
        "cover_cover_channels": channels,

        "cover_mean_intensity": float(
            np.mean(gray)
        ),

        "cover_std_intensity": float(
            np.std(gray)
        ),

        "cover_entropy": calculate_entropy(
            gray
        ),

        "cover_edge_density": calculate_edge_density(
            image
        ),
    }


# ============================================================
# PAYLOAD FEATURES
# ============================================================

def extract_payload_features(payload_path):

    payload_type = detect_payload_type(
        payload_path
    )

    payload_size = payload_path.stat().st_size

    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    features = {
        "payload_type": payload_type,

        "payload_payload_size_bytes":
            payload_size,

        "payload_text_length": np.nan,

        "payload_duration_seconds": np.nan,

        "payload_sample_rate": np.nan,

        "payload_channels": np.nan,

        "payload_sample_width_bytes": np.nan,

        "payload_audio_frames": np.nan,

        "payload_payload_width": np.nan,

        "payload_payload_height": np.nan,

        "payload_payload_channels": np.nan,

        "payload_payload_pixels": np.nan,
    }

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if payload_type == "TEXT":

        try:

            with open(
                payload_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            features[
                "payload_text_length"
            ] = len(text)

        except UnicodeDecodeError:

            with open(
                payload_path,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as file:

                text = file.read()

            features[
                "payload_text_length"
            ] = len(text)

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    elif payload_type == "AUDIO":

        with wave.open(
            str(payload_path),
            "rb"
        ) as audio:

            frames = audio.getnframes()
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()

        duration = (
            frames / sample_rate
            if sample_rate > 0
            else 0
        )

        features[
            "payload_duration_seconds"
        ] = float(duration)

        features[
            "payload_sample_rate"
        ] = sample_rate

        features[
            "payload_channels"
        ] = channels

        features[
            "payload_sample_width_bytes"
        ] = sample_width

        features[
            "payload_audio_frames"
        ] = frames

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    elif payload_type == "IMAGE":

        image = cv2.imread(
            str(payload_path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise ValueError(
                f"Could not read payload image:\n"
                f"{payload_path}"
            )

        if len(image.shape) == 2:

            height, width = image.shape
            channels = 1

        else:

            height, width, channels = (
                image.shape
            )

        features[
            "payload_payload_width"
        ] = width

        features[
            "payload_payload_height"
        ] = height

        features[
            "payload_payload_channels"
        ] = channels

        features[
            "payload_payload_pixels"
        ] = width * height

    return features


# ============================================================
# PAYLOAD HEADER
# ============================================================

def calculate_total_payload_bytes(payload_path):

    """
    AISG payload consists of:

    MAGIC              4 bytes
    VERSION            1 byte
    PAYLOAD TYPE       1 byte
    EXTENSION LENGTH   1 byte
    EXTENSION          variable
    PAYLOAD SIZE       8 bytes
    PAYLOAD DATA       variable

    Therefore:
        total = payload size + 15 + extension length
    """

    extension = payload_path.suffix.encode(
        "utf-8"
    )

    extension_length = len(
        extension
    )

    header_size = (
        4      # MAGIC
        + 1    # VERSION
        + 1    # TYPE
        + 1    # EXTENSION LENGTH
        + extension_length
        + 8    # PAYLOAD SIZE
    )

    return (
        payload_path.stat().st_size
        + header_size
    )


# ============================================================
# CAPACITY
# ============================================================

def get_capacity_bytes(
    method_name,
    cover_path
):

    method_module = METHODS[
        method_name
    ]

    capacity = (
        method_module.calculate_capacity(
            cover_path
        )
    )

    # DWT returns tuple:
    # (capacity_bits, capacity_bytes)

    if isinstance(
        capacity,
        tuple
    ):

        return int(
            capacity[-1]
        )

    # Some implementations return
    # a dictionary.

    if isinstance(
        capacity,
        dict
    ):

        return int(
            capacity.get(
                "capacity_bytes",
                0
            )
        )

    return int(capacity)


# ============================================================
# BUILD CANDIDATE ROW
# ============================================================

def build_candidate_row(
    method_name,
    cover_features,
    payload_features,
    capacity_bytes,
    total_payload_bytes
):

    required_bits = (
        total_payload_bytes * 8
    )

    capacity_fit = (
        required_bits
        <= capacity_bytes * 8
    )

    if capacity_bytes > 0:

        utilization = (
            total_payload_bytes
            / capacity_bytes
            * 100
        )

    else:

        utilization = 0.0

    row = {}

    row.update(
        cover_features
    )

    row.update(
        payload_features
    )

    row.update(
        {
            "method": method_name,

            "capacity_bytes":
                capacity_bytes,

            "required_bits":
                required_bits,

            "capacity_fit_numeric":
                int(capacity_fit),

            "capacity_utilization_percent":
                float(utilization),
        }
    )

    return row


# ============================================================
# AUTO RECOMMEND
# ============================================================

def auto_recommend(
    cover_path,
    payload_path
):

    cover_path = Path(
        cover_path
    )

    payload_path = Path(
        payload_path
    )

    if not cover_path.exists():
        raise FileNotFoundError(
            f"Cover image not found:\n{cover_path}"
        )

    if not payload_path.exists():
        raise FileNotFoundError(
            f"Payload file not found:\n{payload_path}"
        )

    # --------------------------------------------------------
    # Preprocess cover
    # --------------------------------------------------------

    processed_cover = preprocess_cover(
        cover_path
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    cover_features = extract_cover_features(
        processed_cover
    )

    payload_features = extract_payload_features(
        payload_path
    )

    total_payload_bytes = (
        calculate_total_payload_bytes(
            payload_path
        )
    )

    # --------------------------------------------------------
    # Build three candidates
    # --------------------------------------------------------

    candidates = []

    for method_name in METHODS:

        capacity_bytes = get_capacity_bytes(
            method_name,
            processed_cover
        )

        row = build_candidate_row(
            method_name,
            cover_features,
            payload_features,
            capacity_bytes,
            total_payload_bytes
        )

        candidates.append(row)

    # --------------------------------------------------------
    # Predict utility
    # --------------------------------------------------------

    predictions = predict_method_utilities(
        candidates
    )

    # --------------------------------------------------------
    # Add capacity information
    # --------------------------------------------------------

    for index, row in enumerate(candidates):

        predictions.loc[
            index,
            "capacity_bytes"
        ] = row["capacity_bytes"]

        predictions.loc[
            index,
            "required_bits"
        ] = row["required_bits"]

        predictions.loc[
            index,
            "capacity_fit_numeric"
        ] = row[
            "capacity_fit_numeric"
        ]

        predictions.loc[
            index,
            "capacity_utilization_percent"
        ] = row[
            "capacity_utilization_percent"
        ]

    # --------------------------------------------------------
    # Remove infeasible candidates
    # --------------------------------------------------------

    feasible = predictions[
        predictions[
            "capacity_fit_numeric"
        ] == 1
    ].copy()

    # --------------------------------------------------------
    # No method fits
    # --------------------------------------------------------

    if feasible.empty:

        return {
            "recommended_method": None,
            "payload_type":
                payload_features[
                    "payload_type"
                ],
            "payload_size_bytes":
                payload_path.stat().st_size,
            "total_payload_bytes":
                total_payload_bytes,
            "processed_cover":
                processed_cover,
            "candidates":
                predictions,
            "reason":
                "Payload exceeds the capacity of all methods."
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
        "recommended_method":
            best_method,

        "predicted_utility":
            float(best_utility),

        "payload_type":
            payload_features[
                "payload_type"
            ],

        "payload_size_bytes":
            payload_path.stat().st_size,

        "total_payload_bytes":
            total_payload_bytes,

        "processed_cover":
            processed_cover,

        "candidates":
            predictions,

        "reason":
            "Highest predicted utility among feasible methods."
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def print_recommendation(result):

    print("\n" + "=" * 70)
    print("AUTO METHOD RECOMMENDATION")
    print("=" * 70)

    print(
        f"\nPayload type : "
        f"{result['payload_type']}"
    )

    print(
        f"Payload size : "
        f"{result['payload_size_bytes']:,} bytes"
    )

    print(
        f"Total AISG payload : "
        f"{result['total_payload_bytes']:,} bytes"
    )

    print("\nCandidate Methods")
    print("-" * 70)

    candidates = result["candidates"]

    for _, row in candidates.iterrows():

        method = row["method"]

        capacity = int(
            row["capacity_bytes"]
        )

        utilization = (
            row[
                "capacity_utilization_percent"
            ]
        )

        fit = (
            row[
                "capacity_fit_numeric"
            ]
            == 1
        )

        utility = row[
            "predicted_utility"
        ]

        if fit:

            print(
                f"{method:<5} | "
                f"Capacity: {capacity:>10,} B | "
                f"Utilization: {utilization:>7.2f}% | "
                f"Utility: {utility:.4f} | "
                f"FIT"
            )

        else:

            print(
                f"{method:<5} | "
                f"Capacity: {capacity:>10,} B | "
                f"Utilization: {utilization:>7.2f}% | "
                f"NOT FEASIBLE"
            )

    print("-" * 70)

    if result[
        "recommended_method"
    ] is not None:

        print(
            "\nRECOMMENDED METHOD: "
            f"{result['recommended_method']}"
        )

        print(
            f"Predicted utility: "
            f"{result['predicted_utility']:.4f}"
        )

    else:

        print(
            "\nNO FEASIBLE METHOD"
        )

        print(
            result["reason"]
        )

    print("=" * 70)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHANGE THESE TWO PATHS IF YOU WANT TO TEST OTHER FILES
    # --------------------------------------------------------

    cover = (
        PROJECT_ROOT
        / "data"
        / "images"
        / "covers"
        / "cover_09.jpg"
    )

    payload = (
        PROJECT_ROOT
        / "data"
        / "payloads"
        / "variants"
        / "text_small.txt"
    )

    print(
        "Testing AUTO recommendation..."
    )

    result = auto_recommend(
        cover,
        payload
    )

    print_recommendation(
        result
    )