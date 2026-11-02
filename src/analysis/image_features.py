"""
Cover Image Feature Extraction
================================

Extracts statistical and structural features from a cover image
for the ML-based steganography method recommender.

Features:
- Width
- Height
- Channels
- Total pixels
- Mean intensity
- Standard deviation
- Entropy
- Edge density
"""

from pathlib import Path

import cv2
import numpy as np


def calculate_entropy(gray_image):
    """
    Calculate Shannon entropy of a grayscale image.
    """

    histogram = cv2.calcHist(
        [gray_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.flatten()

    # Convert histogram to probabilities
    probabilities = histogram / histogram.sum()

    # Remove zero probabilities
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def calculate_edge_density(gray_image):
    """
    Calculate edge density using the Canny edge detector.

    Edge density =
        Number of edge pixels / Total pixels
    """

    edges = cv2.Canny(
        gray_image,
        threshold1=100,
        threshold2=200
    )

    edge_pixels = np.count_nonzero(edges)

    total_pixels = gray_image.shape[0] * gray_image.shape[1]

    edge_density = edge_pixels / total_pixels

    return float(edge_density)


def extract_image_features(image_path):
    """
    Extract features from a cover image.

    Parameters
    ----------
    image_path : str or Path
        Path to the cover image.

    Returns
    -------
    dict
        Dictionary containing image features.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Read original image
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # Image dimensions
    height, width, channels = image.shape

    total_pixels = height * width

    # Convert to grayscale for statistical analysis
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Mean intensity
    mean_intensity = float(
        np.mean(gray)
    )

    # Standard deviation
    std_intensity = float(
        np.std(gray)
    )

    # Entropy
    entropy = calculate_entropy(gray)

    # Edge density
    edge_density = calculate_edge_density(gray)

    features = {
        "cover_width": width,
        "cover_height": height,
        "cover_channels": channels,
        "cover_pixels": total_pixels,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "entropy": entropy,
        "edge_density": edge_density,
    }

    return features


def print_image_features(features):
    """
    Print extracted features in a readable format.
    """

    print("\n" + "=" * 65)
    print("COVER IMAGE FEATURES")
    print("=" * 65)

    print(f"Width              : {features['cover_width']}")
    print(f"Height             : {features['cover_height']}")
    print(f"Channels           : {features['cover_channels']}")
    print(f"Total pixels       : {features['cover_pixels']:,}")

    print(
        f"Mean intensity     : "
        f"{features['mean_intensity']:.4f}"
    )

    print(
        f"Std. deviation     : "
        f"{features['std_intensity']:.4f}"
    )

    print(
        f"Entropy             : "
        f"{features['entropy']:.4f}"
    )

    print(
        f"Edge density       : "
        f"{features['edge_density']:.6f}"
    )

    print("=" * 65)


if __name__ == "__main__":

    # Current processed cover image
    image_path = (
        "data/images/processed/"
        "cover_processed.png"
    )

    print("Extracting features...")
    print(f"Input image: {image_path}")

    features = extract_image_features(image_path)

    print_image_features(features)

    print("\n✓ Feature extraction successful!")