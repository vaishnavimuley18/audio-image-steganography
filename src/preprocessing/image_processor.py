from PIL import Image
from pathlib import Path


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

INPUT_PATH = Path("data/images/cover.jpeg")
OUTPUT_PATH = Path("data/images/processed/cover_processed.png")

TARGET_SIZE = (4000, 4000)


def preprocess_image():
    print("=" * 55)
    print("IMAGE PREPROCESSING")
    print("=" * 55)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input image not found: {INPUT_PATH}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------
    image = Image.open(INPUT_PATH)

    print(f"Input file       : {INPUT_PATH}")
    print(f"Original format  : {image.format}")
    print(f"Original size    : {image.size}")
    print(f"Original mode    : {image.mode}")

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------
    if image.mode != "RGB":
        image = image.convert("RGB")
        print("Color conversion : Converted to RGB")
    else:
        print("Color conversion : Already RGB")

    original_width, original_height = image.size

    # --------------------------------------------------------
    # Calculate aspect-ratio-preserving dimensions
    # --------------------------------------------------------
    target_width, target_height = TARGET_SIZE

    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    print(
        f"Resizing         : "
        f"{original_width} x {original_height} "
        f"-> {new_width} x {new_height}"
    )

    # --------------------------------------------------------
    # Resize while preserving aspect ratio
    # --------------------------------------------------------
    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Create 4000 x 4000 RGB canvas
    # --------------------------------------------------------
    canvas = Image.new(
        "RGB",
        TARGET_SIZE,
        (0, 0, 0)
    )

    # --------------------------------------------------------
    # Center resized image
    # --------------------------------------------------------
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas.paste(
        resized,
        (x_offset, y_offset)
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save as PNG
    # --------------------------------------------------------
    canvas.save(
        OUTPUT_PATH,
        format="PNG"
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------
    print(f"Padding          : Centered on {target_width} x {target_height}")
    print(f"Output file      : {OUTPUT_PATH}")
    print(f"Output format    : PNG")
    print(f"Output size      : {canvas.size[0]} x {canvas.size[1]}")
    print(f"Output mode      : {canvas.mode}")

    print()
    print("Image preprocessing completed successfully! [OK]")


if __name__ == "__main__":
    preprocess_image()