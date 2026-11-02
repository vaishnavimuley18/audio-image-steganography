"""
Payload Feature Extraction
==========================

Extracts features from TEXT, AUDIO, and IMAGE payloads
for the ML-based steganography method recommender.
"""

from pathlib import Path
import wave

import cv2
import numpy as np


# ============================================================
# TEXT FEATURES
# ============================================================

def extract_text_features(file_path):
    """Extract features from a text payload."""

    file_path = Path(file_path)

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

    file_size = file_path.stat().st_size

    return {
        "payload_type": "TEXT",
        "payload_size_bytes": file_size,
        "payload_size_kb": file_size / 1024,
        "payload_size_mb": file_size / (1024 ** 2),
        "text_length": len(text),
    }


# ============================================================
# AUDIO FEATURES
# ============================================================

def extract_audio_features(file_path):
    """Extract features from a WAV audio payload."""

    file_path = Path(file_path)

    try:
        with wave.open(str(file_path), "rb") as audio:

            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            frames = audio.getnframes()

            duration = frames / sample_rate

    except wave.Error as error:
        raise ValueError(
            f"Unable to read WAV audio: {error}"
        )

    file_size = file_path.stat().st_size

    return {
        "payload_type": "AUDIO",
        "payload_size_bytes": file_size,
        "payload_size_kb": file_size / 1024,
        "payload_size_mb": file_size / (1024 ** 2),
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "audio_frames": frames,
    }


# ============================================================
# IMAGE FEATURES
# ============================================================

def extract_image_payload_features(file_path):
    """Extract features from an image payload."""

    file_path = Path(file_path)

    image = cv2.imread(
        str(file_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Unable to read image payload: {file_path}"
        )

    height, width, channels = image.shape

    file_size = file_path.stat().st_size

    return {
        "payload_type": "IMAGE",
        "payload_size_bytes": file_size,
        "payload_size_kb": file_size / 1024,
        "payload_size_mb": file_size / (1024 ** 2),
        "payload_width": width,
        "payload_height": height,
        "payload_channels": channels,
        "payload_pixels": width * height,
    }


# ============================================================
# GENERAL FUNCTION
# ============================================================

def extract_payload_features(file_path):
    """
    Automatically detect payload type and extract
    the appropriate features.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Payload not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    text_extensions = {
        ".txt",
        ".text",
        ".md",
        ".csv",
        ".json"
    }

    audio_extensions = {
        ".wav"
    }

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    }

    if extension in text_extensions:
        return extract_text_features(file_path)

    elif extension in audio_extensions:
        return extract_audio_features(file_path)

    elif extension in image_extensions:
        return extract_image_payload_features(file_path)

    else:
        raise ValueError(
            f"Unsupported payload extension: {extension}"
        )


# ============================================================
# PRINT FUNCTION
# ============================================================

def print_payload_features(features):
    """Print payload features in a readable format."""

    print("\n" + "=" * 65)
    print("PAYLOAD FEATURES")
    print("=" * 65)

    for key, value in features.items():

        if isinstance(value, float):
            print(f"{key:<25}: {value:.4f}")
        else:
            print(f"{key:<25}: {value}")

    print("=" * 65)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_payloads = [
        "data/payloads/test.txt",
        "data/payloads/test.wav",
        "data/payloads/test.png",
    ]

    print("\n" + "=" * 65)
    print("PAYLOAD FEATURE EXTRACTION TEST")
    print("=" * 65)

    for payload in test_payloads:

        print(f"\nTesting: {payload}")

        try:

            features = extract_payload_features(payload)

            print_payload_features(features)

            print("✓ Extraction successful")

        except Exception as error:

            print(f"✗ ERROR: {error}")

    print("\n" + "=" * 65)
    print("PAYLOAD FEATURE TEST COMPLETE")
    print("=" * 65)