"""
Automated Experimental Results Runner

Runs systematic experiments for:

    LSB
    DCT
    DWT

Payloads:

    TEXT
    AUDIO
    IMAGE

For every experiment the runner records:

    - Method
    - Payload type
    - Payload size
    - Total payload size
    - Cover dimensions
    - Capacity
    - Capacity utilization
    - Capacity fit
    - Embedding time
    - Extraction time
    - Data integrity
    - MSE
    - PSNR
    - SSIM
    - Changed pixels
    - Changed pixel percentage
    - Changed channels
    - Changed channel percentage

Results are saved to:

    results/experimental_results.csv
"""

from pathlib import Path
import csv
import time

import cv2

from src.traditional import lsb
from src.traditional import dct
from src.traditional import dwt

from src.preprocessing.payload_processor import (
    create_payload_from_file
)

from src.utils.payload_format import (
    parse_payload
)

from src.analysis.metrics import (
    calculate_metrics
)


# ============================================================
# PATHS
# ============================================================

COVER_IMAGE = Path(
    "data/images/processed/cover_processed.png"
)

PAYLOADS = [
    Path("data/payloads/test.txt"),
    Path("data/payloads/test.wav"),
    Path("data/payloads/test.png"),
]

OUTPUT_STEGO_DIR = Path(
    "output/stego/experiments"
)

OUTPUT_EXTRACTED_DIR = Path(
    "output/extracted/experiments"
)

RESULTS_DIR = Path(
    "results"
)

RESULTS_FILE = (
    RESULTS_DIR /
    "experimental_results.csv"
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
# RESULT COLUMNS
# ============================================================

RESULT_COLUMNS = [
    "method",
    "payload_type",
    "payload_extension",
    "payload_size_bytes",
    "total_payload_size_bytes",

    "cover_width",
    "cover_height",

    "capacity_bits",
    "capacity_bytes",

    "required_bits",
    "capacity_utilization_percent",
    "capacity_fit",

    "embedding_time_seconds",
    "extraction_time_seconds",

    "integrity_percent",

    "mse",
    "psnr",
    "ssim",

    "changed_pixels",
    "total_pixels",
    "changed_pixel_percentage",

    "changed_channels",
    "total_channels",
    "changed_channel_percentage",

    "status",
    "error",
]


# ============================================================
# HELPERS
# ============================================================

def get_payload_info(payload_path):
    """
    Read the payload and determine its common AISG information.
    """

    payload = create_payload_from_file(
        payload_path
    )

    info = parse_payload(
        payload
    )

    return {
        "payload_type": info["payload_type"],
        "extension": info["extension"],
        "payload_size": info["size"],
        "total_payload_size": len(payload),
    }


def get_cover_dimensions():
    """
    Get cover image dimensions.
    """

    image = cv2.imread(
        str(COVER_IMAGE),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise FileNotFoundError(
            f"Cover image not found: {COVER_IMAGE}"
        )

    height, width = image.shape[:2]

    return width, height


def get_method_capacity(
    method_name
):
    """
    Get capacity from the corresponding
    steganography implementation.
    """

    method = METHODS[method_name]

    capacity = method.calculate_capacity(
        str(COVER_IMAGE)
    )

    if method_name == "LSB":

        return (
            capacity["capacity_bits"],
            capacity["capacity_bytes"]
        )

    elif method_name == "DCT":

        return (
            capacity["capacity_bits"],
            capacity["capacity_bytes"]
        )

    elif method_name == "DWT":

        capacity_bits, capacity_bytes = capacity

        return (
            capacity_bits,
            capacity_bytes
        )

    else:

        raise ValueError(
            f"Unsupported method: {method_name}"
        )


def load_image_for_metrics(
    method_name,
    image_path
):
    """
    Load images in a consistent manner for
    the current implementation.

    LSB:
        RGB

    DCT:
        Grayscale

    DWT:
        Grayscale
    """

    if method_name == "LSB":

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

    else:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def verify_integrity(
    original_payload_path,
    extracted_path
):
    """
    Compare original and extracted payload bytes.
    """

    original_data = Path(
        original_payload_path
    ).read_bytes()

    extracted_data = Path(
        extracted_path
    ).read_bytes()

    if original_data == extracted_data:

        return 100.0

    return 0.0


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_single_experiment(
    method_name,
    payload_path
):
    """
    Run one complete experiment.

    Example:

        LSB + TEXT
        DCT + AUDIO
        DWT + IMAGE
    """

    method = METHODS[method_name]

    payload_path = Path(
        payload_path
    )

    print("\n")
    print("#" * 75)
    print(
        f"EXPERIMENT: {method_name} + "
        f"{payload_path.name}"
    )
    print("#" * 75)

    # --------------------------------------------------------
    # Payload information
    # --------------------------------------------------------

    payload_info = get_payload_info(
        payload_path
    )

    payload_type = payload_info[
        "payload_type"
    ]

    payload_extension = payload_info[
        "extension"
    ]

    payload_size = payload_info[
        "payload_size"
    ]

    total_payload_size = payload_info[
        "total_payload_size"
    ]

    required_bits = (
        total_payload_size * 8
    )

    # --------------------------------------------------------
    # Cover dimensions
    # --------------------------------------------------------

    width, height = (
        get_cover_dimensions()
    )

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    capacity_bits, capacity_bytes = (
        get_method_capacity(
            method_name
        )
    )

    utilization = (
        required_bits /
        capacity_bits *
        100
    )

    capacity_fit = (
        required_bits <= capacity_bits
    )

    print(
        f"Method              : {method_name}"
    )

    print(
        f"Payload type        : {payload_type}"
    )

    print(
        f"Payload size        : "
        f"{payload_size:,} bytes"
    )

    print(
        f"Total payload       : "
        f"{total_payload_size:,} bytes"
    )

    print(
        f"Capacity            : "
        f"{capacity_bytes:,} bytes"
    )

    print(
        f"Required bits       : "
        f"{required_bits:,}"
    )

    print(
        f"Utilization         : "
        f"{utilization:.4f}%"
    )

    print(
        f"Capacity fit        : "
        f"{capacity_fit}"
    )

    # --------------------------------------------------------
    # Base result
    # --------------------------------------------------------

    result = {
        "method": method_name,

        "payload_type": payload_type,

        "payload_extension":
            payload_extension,

        "payload_size_bytes":
            payload_size,

        "total_payload_size_bytes":
            total_payload_size,

        "cover_width":
            width,

        "cover_height":
            height,

        "capacity_bits":
            capacity_bits,

        "capacity_bytes":
            capacity_bytes,

        "required_bits":
            required_bits,

        "capacity_utilization_percent":
            utilization,

        "capacity_fit":
            capacity_fit,

        "embedding_time_seconds":
            None,

        "extraction_time_seconds":
            None,

        "integrity_percent":
            None,

        "mse":
            None,

        "psnr":
            None,

        "ssim":
            None,

        "changed_pixels":
            None,

        "total_pixels":
            None,

        "changed_pixel_percentage":
            None,

        "changed_channels":
            None,

        "total_channels":
            None,

        "changed_channel_percentage":
            None,

        "status":
            None,

        "error":
            "",
    }

    # --------------------------------------------------------
    # If payload does not fit
    # --------------------------------------------------------

    if not capacity_fit:

        result["status"] = (
            "INSUFFICIENT_CAPACITY"
        )

        result["error"] = (
            f"Required {required_bits:,} bits; "
            f"available {capacity_bits:,} bits."
        )

        print(
            "\nSTATUS: "
            "INSUFFICIENT CAPACITY ❌"
        )

        return result

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    safe_payload_name = payload_path.stem.lower()
    payload_extension = payload_path.suffix.lower().lstrip(".")

    experiment_name = f"{safe_payload_name}_{payload_extension}"

    stego_path = (
        OUTPUT_STEGO_DIR /
        f"{method_name.lower()}_{experiment_name}.png"
    )

    extraction_directory = (
        OUTPUT_EXTRACTED_DIR /
        f"{method_name.lower()}_{experiment_name}"
    )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    try:

        embed_result = (
            method.embed_payload(
                str(COVER_IMAGE),
                str(payload_path),
                str(stego_path)
            )
        )

        result["embedding_time_seconds"] = (
            embed_result.get(
                "embedding_time",
                None
            )
        )

        # DWT uses "utilization" while
        # LSB/DCT use "utilization_percent".
        if (
            "utilization_percent"
            in embed_result
        ):

            result[
                "capacity_utilization_percent"
            ] = embed_result[
                "utilization_percent"
            ]

        elif (
            "utilization"
            in embed_result
        ):

            result[
                "capacity_utilization_percent"
            ] = embed_result[
                "utilization"
            ]

    except Exception as error:

        result["status"] = (
            "EMBEDDING_FAILED"
        )

        result["error"] = str(
            error
        )

        print(
            "\nSTATUS: "
            "EMBEDDING FAILED ❌"
        )

        print(
            f"Error: {error}"
        )

        return result

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    try:

        cover_for_metrics = (
            load_image_for_metrics(
                method_name,
                COVER_IMAGE
            )
        )

        stego_for_metrics = (
            load_image_for_metrics(
                method_name,
                stego_path
            )
        )

        print(
            f"\nMetric images:"
            f" cover={cover_for_metrics.shape},"
            f" stego={stego_for_metrics.shape}"
        )

        metrics = calculate_metrics(
            cover_for_metrics,
            stego_for_metrics
        )

        result["mse"] = float(
            metrics["mse"]
        )

        result["psnr"] = float(
            metrics["psnr"]
        )

        result["ssim"] = float(
            metrics["ssim"]
        )

        result["changed_pixels"] = int(
            metrics["changed_pixels"]
        )

        result["total_pixels"] = int(
            metrics["total_pixels"]
        )

        result[
            "changed_pixel_percentage"
        ] = float(
            metrics["changed_percentage"]
        )

        result["changed_channels"] = int(
            metrics["changed_channels"]
        )

        result["total_channels"] = int(
            metrics["total_channels"]
        )

        result[
            "changed_channel_percentage"
        ] = float(
            metrics[
                "changed_channel_percentage"
            ]
        )

    except Exception as error:

        print(
            "\nWARNING: "
            "Metric calculation failed ⚠️"
        )

        print(
            f"Metric error: {error}"
        )

    # --------------------------------------------------------
    # EXTRACTION
    # --------------------------------------------------------

    try:

        extract_result = (
            method.extract_payload(
                str(stego_path),
                str(extraction_directory)
            )
        )

        result[
            "extraction_time_seconds"
        ] = extract_result.get(
            "extraction_time",
            None
        )

        extracted_path = (
            extract_result[
                "output_path"
            ]
        )

        # ----------------------------------------------------
        # Integrity
        # ----------------------------------------------------

        integrity = verify_integrity(
            payload_path,
            extracted_path
        )

        result[
            "integrity_percent"
        ] = integrity

        if integrity == 100.0:

            result["status"] = (
                "SUCCESS"
            )

            print(
                "\nSTATUS: "
                "SUCCESS ✅"
            )

        else:

            result["status"] = (
                "INTEGRITY_FAILED"
            )

            result["error"] = (
                "Extracted payload "
                "does not match original."
            )

            print(
                "\nSTATUS: "
                "INTEGRITY FAILED ❌"
            )

    except Exception as error:

        result["status"] = (
            "EXTRACTION_FAILED"
        )

        result["error"] = str(
            error
        )

        print(
            "\nSTATUS: "
            "EXTRACTION FAILED ❌"
        )

        print(
            f"Error: {error}"
        )

    return result


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):
    """
    Save experiment results to CSV.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_COLUMNS
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print("\n")
    print("=" * 75)
    print("RESULTS SAVED")
    print("=" * 75)
    print(
        f"File: {RESULTS_FILE}"
    )
    print(
        f"Experiments: {len(results)}"
    )
    print("=" * 75)


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):
    """
    Print a compact experiment summary.
    """

    print("\n")
    print("=" * 75)
    print("EXPERIMENT SUMMARY")
    print("=" * 75)

    print(
        f"{'METHOD':<8}"
        f"{'PAYLOAD':<10}"
        f"{'FIT':<10}"
        f"{'STATUS':<25}"
        f"{'PSNR':<12}"
        f"{'SSIM':<10}"
    )

    print("-" * 75)

    for result in results:

        psnr = result["psnr"]

        ssim = result["ssim"]

        psnr_text = (
            f"{psnr:.2f}"
            if psnr is not None
            else "-"
        )

        ssim_text = (
            f"{ssim:.4f}"
            if ssim is not None
            else "-"
        )

        print(
            f"{result['method']:<8}"
            f"{result['payload_type']:<10}"
            f"{str(result['capacity_fit']):<10}"
            f"{result['status']:<25}"
            f"{psnr_text:<12}"
            f"{ssim_text:<10}"
        )

    print("=" * 75)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_all_experiments():
    """
    Run every method against every payload.

    Total experiments:

        3 methods × 3 payloads

        = 9 experiments
    """

    print("\n")
    print("=" * 75)
    print("AUTOMATED STEGANOGRAPHY EXPERIMENTS")
    print("=" * 75)

    print(
        f"Cover image: {COVER_IMAGE}"
    )

    print(
        f"Methods    : {', '.join(METHODS.keys())}"
    )

    print(
        f"Payloads   : {len(PAYLOADS)}"
    )

    print(
        "Total experiments: "
        f"{len(METHODS) * len(PAYLOADS)}"
    )

    print("=" * 75)

    results = []

    overall_start = time.perf_counter()

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for method_name in METHODS:

        for payload_path in PAYLOADS:

            result = run_single_experiment(
                method_name,
                payload_path
            )

            results.append(
                result
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    overall_time = (
        time.perf_counter() -
        overall_start
    )

    print(
        f"\nTotal experiment time: "
        f"{overall_time:.4f} seconds"
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_all_experiments()