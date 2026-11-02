from pathlib import Path
import contextlib
import csv
import io
import shutil
import time

import cv2
from PIL import Image

from src.traditional import lsb, dct, dwt
from src.analysis.metrics import calculate_metrics


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COVER_DIR = PROJECT_ROOT / "data" / "images" / "covers"
PAYLOAD_DIR = PROJECT_ROOT / "data" / "payloads" / "variants"

PROCESSED_COVER_DIR = (
    PROJECT_ROOT / "data" / "images" / "processed" / "multi_cover_test"
)

OUTPUT_STEGO_DIR = (
    PROJECT_ROOT / "output" / "stego" / "multi_cover_test"
)

OUTPUT_EXTRACTED_DIR = (
    PROJECT_ROOT / "output" / "extracted" / "multi_cover_test"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "multi_cover_results.csv"


# ============================================================
# TEST CONFIGURATION
# ============================================================

METHODS = {
    "LSB": lsb,
    "DCT": dct,
    "DWT": dwt,
}

PAYLOAD_FILES = [
    PAYLOAD_DIR / "text_small.txt",
    PAYLOAD_DIR / "text_medium.txt",
    PAYLOAD_DIR / "text_large.txt",

    PAYLOAD_DIR / "audio_small.wav",
    PAYLOAD_DIR / "audio_medium.wav",
    PAYLOAD_DIR / "audio_large.wav",

    PAYLOAD_DIR / "image_small.png",
    PAYLOAD_DIR / "image_medium.png",
    PAYLOAD_DIR / "image_large.png",
]


# ============================================================
# COVER PREPROCESSING
# ============================================================

def preprocess_cover(input_path, output_path):
    """Convert cover to RGB and resize/pad to exactly 4000x4000."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        image = image.convert("RGB")

        target_size = 4000

        scale = min(
            target_size / image.width,
            target_size / image.height,
        )

        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (target_size, target_size),
            (0, 0, 0),
        )

        x = (target_size - new_width) // 2
        y = (target_size - new_height) // 2

        canvas.paste(resized, (x, y))
        canvas.save(output_path, format="PNG")

    return output_path


# ============================================================
# RESULT ROW
# ============================================================

def empty_result():
    return {
        "cover_name": "",
        "method": "",
        "payload_name": "",
        "payload_type": "",
        "payload_size_bytes": "",
        "stego_path": "",
        "capacity_bytes": "",
        "required_bits": "",
        "capacity_utilization_percent": "",
        "capacity_fit": "",
        "embedding_time_seconds": "",
        "extraction_time_seconds": "",
        "integrity_percent": "",
        "mse": "",
        "psnr": "",
        "ssim": "",
        "changed_pixels": "",
        "total_pixels": "",
        "changed_pixel_percentage": "",
        "changed_channels": "",
        "total_channels": "",
        "changed_channel_percentage": "",
        "status": "",
        "error": "",
    }


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    method_name,
    method_module,
    cover_path,
    payload_path,
    output_stego_path,
    extraction_directory,
):
    result = empty_result()

    result["cover_name"] = cover_path.name
    result["method"] = method_name
    result["payload_name"] = payload_path.name
    result["stego_path"] = str(output_stego_path)

    payload_size = payload_path.stat().st_size
    result["payload_size_bytes"] = payload_size

    extension = payload_path.suffix.lower()

    if extension == ".txt":
        payload_type = "TEXT"
    elif extension == ".wav":
        payload_type = "AUDIO"
    elif extension in [".png", ".jpg", ".jpeg"]:
        payload_type = "IMAGE"
    else:
        payload_type = "UNKNOWN"

    result["payload_type"] = payload_type

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    try:
        capacity = method_module.calculate_capacity(cover_path)

        if isinstance(capacity, dict):
            capacity_bytes = int(
                capacity.get("capacity_bytes", 0)
            )
        elif isinstance(capacity, tuple):
            capacity_bytes = int(capacity[-1])
        else:
            capacity_bytes = int(capacity)

        result["capacity_bytes"] = capacity_bytes

        # Common AISG header = 18 bytes.
        total_payload_size = payload_size + 18
        required_bits = total_payload_size * 8

        result["required_bits"] = required_bits

        result["capacity_utilization_percent"] = (
            total_payload_size / capacity_bytes * 100
            if capacity_bytes > 0
            else 0
        )

        fits = total_payload_size <= capacity_bytes
        result["capacity_fit"] = fits

        if not fits:
            result["status"] = "INSUFFICIENT_CAPACITY"
            return result

    except Exception as e:
        result["status"] = "CAPACITY_ERROR"
        result["error"] = str(e)
        return result

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    try:
        output_stego_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        extraction_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        start_time = time.perf_counter()

        # The traditional modules print detailed logs.
        # Redirect them so the console remains compact.
        with contextlib.redirect_stdout(io.StringIO()):
            method_module.embed_payload(
                cover_path,
                payload_path,
                output_stego_path,
            )

        result["embedding_time_seconds"] = (
            time.perf_counter() - start_time
        )

    except Exception as e:
        result["status"] = "EMBEDDING_FAILED"
        result["error"] = str(e)
        return result

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    try:
        start_time = time.perf_counter()

        with contextlib.redirect_stdout(io.StringIO()):
            extraction_result = method_module.extract_payload(
                output_stego_path,
                extraction_directory,
            )

        result["extraction_time_seconds"] = (
            time.perf_counter() - start_time
        )

        extracted_file = None

        if isinstance(extraction_result, Path):
            extracted_file = extraction_result
        elif isinstance(extraction_result, str):
            extracted_file = Path(extraction_result)

        if (
            extracted_file is None
            or not extracted_file.exists()
        ):
            candidates = list(extraction_directory.iterdir())

            if candidates:
                extracted_file = candidates[0]

        if (
            extracted_file is not None
            and extracted_file.exists()
        ):
            original_bytes = payload_path.read_bytes()
            extracted_bytes = extracted_file.read_bytes()

            if original_bytes == extracted_bytes:
                integrity = 100.0
            else:
                matching = sum(
                    a == b
                    for a, b in zip(
                        original_bytes,
                        extracted_bytes,
                    )
                )

                integrity = (
                    matching /
                    max(len(original_bytes), 1)
                    * 100
                )

            result["integrity_percent"] = integrity
        else:
            result["integrity_percent"] = 0.0

    except Exception as e:
        result["status"] = "EXTRACTION_FAILED"
        result["error"] = str(e)
        return result

    # --------------------------------------------------------
    # Image metrics
    # --------------------------------------------------------

    try:
        if method_name == "LSB":
            cover_image = cv2.imread(
                str(cover_path),
                cv2.IMREAD_COLOR,
            )

            stego_image = cv2.imread(
                str(output_stego_path),
                cv2.IMREAD_COLOR,
            )
        else:
            cover_image = cv2.imread(
                str(cover_path),
                cv2.IMREAD_GRAYSCALE,
            )

            stego_image = cv2.imread(
                str(output_stego_path),
                cv2.IMREAD_GRAYSCALE,
            )

        if cover_image is None:
            raise FileNotFoundError(
                f"Could not load cover image: {cover_path}"
            )

        if stego_image is None:
            raise FileNotFoundError(
                f"Could not load stego image: {output_stego_path}"
            )

        metrics = calculate_metrics(
            cover_image,
            stego_image,
        )

        result["mse"] = metrics["mse"]
        result["psnr"] = metrics["psnr"]
        result["ssim"] = metrics["ssim"]

        result["changed_pixels"] = metrics["changed_pixels"]
        result["total_pixels"] = metrics["total_pixels"]
        result["changed_pixel_percentage"] = (
            metrics["changed_percentage"]
        )

        result["changed_channels"] = metrics["changed_channels"]
        result["total_channels"] = metrics["total_channels"]
        result["changed_channel_percentage"] = (
            metrics["changed_channel_percentage"]
        )

    except Exception as e:
        result["status"] = "METRICS_FAILED"
        result["error"] = str(e)
        return result

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if result["integrity_percent"] == 100.0:
        result["status"] = "SUCCESS"
    else:
        result["status"] = "INTEGRITY_FAILED"

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("MULTI-COVER EXPERIMENT")
    print("=" * 60)

    # Discover every cover_*.jpg automatically.
    cover_files = sorted(
        COVER_DIR.glob("cover_*.jpg")
    )

    if not cover_files:
        raise FileNotFoundError(
            f"No cover images found in:\n{COVER_DIR}"
        )

    for payload in PAYLOAD_FILES:
        if not payload.exists():
            raise FileNotFoundError(
                f"Payload not found:\n{payload}"
            )

    total_experiments = (
        len(cover_files)
        * len(PAYLOAD_FILES)
        * len(METHODS)
    )

    print(f"Covers       : {len(cover_files)}")
    print(f"Payloads     : {len(PAYLOAD_FILES)}")
    print(f"Methods      : {len(METHODS)}")
    print(f"Experiments  : {total_experiments}")
    print("\nRunning experiments...")

    PROCESSED_COVER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_STEGO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_EXTRACTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # --------------------------------------------------------
    # All covers
    # --------------------------------------------------------

    for cover_index, cover_path in enumerate(
        cover_files,
        start=1,
    ):
        processed_cover = (
            PROCESSED_COVER_DIR
            / f"{cover_path.stem}_processed.png"
        )

        preprocess_cover(
            cover_path,
            processed_cover,
        )

        cover_results_start = len(results)

        for payload_path in PAYLOAD_FILES:

            payload_name = payload_path.stem

            for method_name, method_module in METHODS.items():

                stego_path = (
                    OUTPUT_STEGO_DIR
                    / (
                        f"{cover_path.stem}_"
                        f"{method_name.lower()}_"
                        f"{payload_name}.png"
                    )
                )

                extraction_dir = (
                    OUTPUT_EXTRACTED_DIR
                    / (
                        f"{cover_path.stem}_"
                        f"{method_name.lower()}_"
                        f"{payload_name}"
                    )
                )

                if stego_path.exists():
                    stego_path.unlink()

                if extraction_dir.exists():
                    shutil.rmtree(extraction_dir)

                result = run_experiment(
                    method_name,
                    method_module,
                    processed_cover,
                    payload_path,
                    stego_path,
                    extraction_dir,
                )

                results.append(result)

        # ----------------------------------------------------
        # Compact progress
        # ----------------------------------------------------

        cover_results = results[cover_results_start:]

        cover_success = sum(
            r["status"] == "SUCCESS"
            for r in cover_results
        )

        cover_capacity_failures = sum(
            r["status"] == "INSUFFICIENT_CAPACITY"
            for r in cover_results
        )

        cover_integrity_failures = sum(
            r["status"] == "INTEGRITY_FAILED"
            for r in cover_results
        )

        cover_other_failures = len(cover_results) - (
            cover_success
            + cover_capacity_failures
            + cover_integrity_failures
        )

        other_text = (
            f"  |  Other: {cover_other_failures}"
            if cover_other_failures
            else ""
        )

        print(
            f"Cover {cover_index:02d}/{len(cover_files):02d}  "
            f"✓  Success: {cover_success}  |  "
            f"Capacity: {cover_capacity_failures}  |  "
            f"Integrity: {cover_integrity_failures}"
            f"{other_text}"
        )

    # --------------------------------------------------------
    # Save and final summary
    # --------------------------------------------------------

    save_results(results)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    success_count = sum(
        r["status"] == "SUCCESS"
        for r in results
    )

    capacity_failures = sum(
        r["status"] == "INSUFFICIENT_CAPACITY"
        for r in results
    )

    integrity_failures = sum(
        r["status"] == "INTEGRITY_FAILED"
        for r in results
    )

    other_failures = len(results) - (
        success_count
        + capacity_failures
        + integrity_failures
    )

    print(f"Total experiments      : {len(results)}")
    print(f"Successful             : {success_count}")
    print(f"Capacity failures      : {capacity_failures}")
    print(f"Integrity failures     : {integrity_failures}")

    if other_failures:
        print(f"Other failures         : {other_failures}")

    print(f"\nResults saved to:\n{RESULTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
