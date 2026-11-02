from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# PATHS
# ============================================================

COVER_PATH = Path(
    "data/images/processed/cover_processed.png"
)

STEGO_PATH = Path(
    "output/stego/lsb_stego.png"
)

DIFFERENCE_PATH = Path(
    "output/analysis/lsb_difference.png"
)


# ============================================================
# DIRECTORY
# ============================================================

DIFFERENCE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD IMAGES
# ============================================================

def load_image(path, grayscale=False):

    image = Image.open(path)

    if grayscale:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    return np.array(image, dtype=np.uint8)


# ============================================================
# IMAGE METRICS
# ============================================================

def calculate_metrics(cover, stego):

    # --------------------------------------------------------
    # Check image shapes
    # --------------------------------------------------------

    if cover.shape != stego.shape:
        raise ValueError(
            "Cover and stego images must have the same shape."
        )

    # Convert to float for calculations
    cover_float = cover.astype(np.float64)
    stego_float = stego.astype(np.float64)

    # --------------------------------------------------------
    # Determine image type
    # --------------------------------------------------------

    is_rgb = (
        cover.ndim == 3
        and cover.shape[2] == 3
    )

    is_grayscale = cover.ndim == 2

    if not (is_rgb or is_grayscale):
        raise ValueError(
            f"Unsupported image shape: {cover.shape}"
        )

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = mean_squared_error(
        cover_float,
        stego_float
    )

    # --------------------------------------------------------
    # PSNR
    # --------------------------------------------------------

    psnr = peak_signal_noise_ratio(
        cover_float,
        stego_float,
        data_range=255
    )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    if is_rgb:

        ssim = structural_similarity(
            cover_float,
            stego_float,
            channel_axis=2,
            data_range=255
        )

    else:

        ssim = structural_similarity(
            cover_float,
            stego_float,
            data_range=255
        )

    # --------------------------------------------------------
    # Pixel differences
    # --------------------------------------------------------

    difference = np.abs(
        cover_float - stego_float
    )

    max_difference = np.max(
        difference
    )

    mean_absolute_difference = np.mean(
        difference
    )

    # --------------------------------------------------------
    # Changed pixels
    # --------------------------------------------------------

    if is_rgb:

        # A pixel is changed if at least one
        # RGB channel is different.

        changed_pixels = np.any(
            cover != stego,
            axis=2
        )

        number_changed_pixels = np.sum(
            changed_pixels
        )

        total_pixels = (
            cover.shape[0] *
            cover.shape[1]
        )

    else:

        # For grayscale, each array element
        # represents one pixel.

        changed_pixels = (
            cover != stego
        )

        number_changed_pixels = np.sum(
            changed_pixels
        )

        total_pixels = (
            cover.shape[0] *
            cover.shape[1]
        )

    changed_percentage = (
        number_changed_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # Changed channels / values
    # --------------------------------------------------------

    changed_channels = np.sum(
        cover != stego
    )

    total_channels = cover.size

    changed_channel_percentage = (
        changed_channels /
        total_channels
    ) * 100

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,

        "max_difference":
            max_difference,

        "mean_absolute_difference":
            mean_absolute_difference,

        "changed_pixels":
            number_changed_pixels,

        "total_pixels":
            total_pixels,

        "changed_percentage":
            changed_percentage,

        "changed_channels":
            changed_channels,

        "total_channels":
            total_channels,

        "changed_channel_percentage":
            changed_channel_percentage
    }


# ============================================================
# DIFFERENCE IMAGE
# ============================================================

def create_difference_image(
    cover,
    stego,
    output_path=DIFFERENCE_PATH
):

    difference = np.abs(
        cover.astype(np.int16) -
        stego.astype(np.int16)
    )

    # Amplify differences so that
    # small changes become visible.
    difference = difference * 50

    difference = np.clip(
        difference,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Save according to image type
    # --------------------------------------------------------

    if difference.ndim == 3:

        Image.fromarray(
            difference,
            mode="RGB"
        ).save(
            output_path
        )

    else:

        Image.fromarray(
            difference,
            mode="L"
        ).save(
            output_path
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IMAGE QUALITY ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not COVER_PATH.exists():

        print("\nERROR: Cover image not found:")
        print(COVER_PATH)
        return

    if not STEGO_PATH.exists():

        print("\nERROR: Stego image not found:")
        print(STEGO_PATH)
        return

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    cover = load_image(
        COVER_PATH,
        grayscale=False
    )

    stego = load_image(
        STEGO_PATH,
        grayscale=False
    )

    print("\nINPUT IMAGES")
    print("-" * 60)

    print(
        f"Cover image      : "
        f"{COVER_PATH}"
    )

    print(
        f"Stego image      : "
        f"{STEGO_PATH}"
    )

    print(
        f"Cover shape      : "
        f"{cover.shape}"
    )

    print(
        f"Stego shape      : "
        f"{stego.shape}"
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if cover.shape != stego.shape:

        print("\nERROR ❌")
        print(
            "Cover and stego images "
            "have different dimensions."
        )
        return

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    results = calculate_metrics(
        cover,
        stego
    )

    # --------------------------------------------------------
    # Display metrics
    # --------------------------------------------------------

    print("\nIMAGE QUALITY METRICS")
    print("-" * 60)

    print(
        f"MSE                : "
        f"{results['mse']:.10f}"
    )

    print(
        f"PSNR               : "
        f"{results['psnr']:.4f} dB"
    )

    print(
        f"SSIM               : "
        f"{results['ssim']:.10f}"
    )

    print(
        f"Maximum difference : "
        f"{results['max_difference']:.0f}"
    )

    print(
        f"Mean absolute diff : "
        f"{results['mean_absolute_difference']:.10f}"
    )

    print("\nPIXEL CHANGES")
    print("-" * 60)

    print(
        f"Total pixels       : "
        f"{results['total_pixels']:,}"
    )

    print(
        f"Changed pixels     : "
        f"{results['changed_pixels']:,}"
    )

    print(
        f"Changed pixels     : "
        f"{results['changed_percentage']:.4f}%"
    )

    print(
        f"Total channels     : "
        f"{results['total_channels']:,}"
    )

    print(
        f"Changed channels   : "
        f"{results['changed_channels']:,}"
    )

    print(
        f"Changed channels   : "
        f"{results['changed_channel_percentage']:.4f}%"
    )

    # --------------------------------------------------------
    # Difference image
    # --------------------------------------------------------

    create_difference_image(
        cover,
        stego
    )

    print("\nDIFFERENCE IMAGE")
    print("-" * 60)

    print(
        f"Output             : "
        f"{DIFFERENCE_PATH}"
    )

    print("\n" + "=" * 60)
    print("IMAGE QUALITY ANALYSIS COMPLETE ✅")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
