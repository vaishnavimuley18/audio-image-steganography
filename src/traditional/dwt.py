"""
DWT Steganography Engine

Method:
    - Integer Haar DWT
    - LH + HL sub-bands
    - 1 bit per coefficient
    - Reversible integer transform
    - Generic AISG payload format
    - Supports TEXT, AUDIO and IMAGE

Rule:
    If payload fits -> embed.
    If payload does not fit -> report insufficient capacity.
"""

import os
import time

import cv2
import numpy as np

from src.preprocessing.payload_processor import create_payload_from_file
from src.utils.payload_format import parse_payload


# ============================================================
# CONFIGURATION
# ============================================================

WAVELET = "haar"


# ============================================================
# INTEGER HAAR 1D TRANSFORM
# ============================================================

def haar_forward_1d(x):
    """
    Reversible integer Haar transform.

    For every pair:

        even = x[0], x[2], ...
        odd  = x[1], x[3], ...

        detail = odd - even
        approx = even + floor(detail / 2)

    This transform is reversible using integers.
    """

    x = np.asarray(
        x,
        dtype=np.int64
    )

    if x.size % 2 != 0:
        raise ValueError(
            "Integer Haar DWT requires even dimensions."
        )

    even = x[::2]
    odd = x[1::2]

    detail = (
        odd - even
    )

    approx = (
        even +
        np.floor_divide(
            detail,
            2
        )
    )

    return approx, detail


def haar_inverse_1d(
    approx,
    detail
):
    """
    Exact inverse of haar_forward_1d().
    """

    approx = np.asarray(
        approx,
        dtype=np.int64
    )

    detail = np.asarray(
        detail,
        dtype=np.int64
    )

    even = (
        approx -
        np.floor_divide(
            detail,
            2
        )
    )

    odd = (
        detail +
        even
    )

    output = np.empty(
        even.size * 2,
        dtype=np.int64
    )

    output[::2] = even
    output[1::2] = odd

    return output


# ============================================================
# 2D INTEGER HAAR DWT
# ============================================================

def apply_dwt(image):
    """
    Apply one-level 2D reversible integer Haar DWT.

    Returns:

        LL
        LH
        HL
        HH

    LH = low vertical + high horizontal
    HL = high vertical + low horizontal
    HH = high vertical + high horizontal
    """

    if image is None:
        raise ValueError(
            "Image could not be loaded."
        )

    if len(image.shape) != 2:
        raise ValueError(
            "DWT expects a grayscale image."
        )

    height, width = image.shape

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            "Image width and height must be even."
        )

    image = image.astype(
        np.int64
    )

    # --------------------------------------------------------
    # Horizontal transform
    # --------------------------------------------------------

    row_approx = np.empty(
        (height, width // 2),
        dtype=np.int64
    )

    row_detail = np.empty(
        (height, width // 2),
        dtype=np.int64
    )

    for row in range(height):

        approx, detail = haar_forward_1d(
            image[row, :]
        )

        row_approx[row, :] = approx
        row_detail[row, :] = detail

    # --------------------------------------------------------
    # Vertical transform
    # --------------------------------------------------------

    half_height = height // 2
    half_width = width // 2

    LL = np.empty(
        (half_height, half_width),
        dtype=np.int64
    )

    LH = np.empty(
        (half_height, half_width),
        dtype=np.int64
    )

    HL = np.empty(
        (half_height, half_width),
        dtype=np.int64
    )

    HH = np.empty(
        (half_height, half_width),
        dtype=np.int64
    )

    for col in range(half_width):

        # Low-horizontal component
        approx, detail = haar_forward_1d(
            row_approx[:, col]
        )

        LL[:, col] = approx
        HL[:, col] = detail

        # High-horizontal component
        approx, detail = haar_forward_1d(
            row_detail[:, col]
        )

        LH[:, col] = approx
        HH[:, col] = detail

    return LL, LH, HL, HH


# ============================================================
# INTEGER HAAR INVERSE
# ============================================================

def apply_idwt(
    LL,
    LH,
    HL,
    HH
):
    """
    Exact inverse of apply_dwt().
    """

    half_height, half_width = LL.shape

    height = (
        half_height * 2
    )

    width = (
        half_width * 2
    )

    # --------------------------------------------------------
    # Inverse vertical transform
    # --------------------------------------------------------

    row_approx = np.empty(
        (height, half_width),
        dtype=np.int64
    )

    row_detail = np.empty(
        (height, half_width),
        dtype=np.int64
    )

    for col in range(half_width):

        row_approx[:, col] = haar_inverse_1d(
            LL[:, col],
            HL[:, col]
        )

        row_detail[:, col] = haar_inverse_1d(
            LH[:, col],
            HH[:, col]
        )

    # --------------------------------------------------------
    # Inverse horizontal transform
    # --------------------------------------------------------

    image = np.empty(
        (height, width),
        dtype=np.int64
    )

    for row in range(height):

        image[row, :] = haar_inverse_1d(
            row_approx[row, :],
            row_detail[row, :]
        )

    return image


# ============================================================
# CAPACITY
# ============================================================

def calculate_capacity(image_path):
    """
    Calculate DWT payload capacity.

    One bit is stored in every coefficient
    of LH and HL.

    For a 4000 x 4000 image:

        LH = 2000 x 2000
        HL = 2000 x 2000

        Total = 8,000,000 coefficients

        Capacity = 1,000,000 bytes
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    _, LH, HL, _ = apply_dwt(
        image
    )

    capacity_bits = (
        LH.size +
        HL.size
    )

    capacity_bytes = (
        capacity_bits // 8
    )

    return (
        capacity_bits,
        capacity_bytes
    )


# ============================================================
# BITS <-> BYTES
# ============================================================

def bytes_to_bits(data):
    """
    Convert bytes into bits.

    Most-significant bit first.
    """

    bits = []

    for byte in data:

        for i in range(
            7,
            -1,
            -1
        ):

            bits.append(
                (byte >> i) & 1
            )

    return bits


def bits_to_bytes(bits):
    """
    Convert bits back into bytes.
    """

    if len(bits) % 8 != 0:
        raise ValueError(
            "Number of bits must be divisible by 8."
        )

    output = bytearray()

    for i in range(
        0,
        len(bits),
        8
    ):

        byte = 0

        for bit in bits[i:i + 8]:

            byte = (
                (byte << 1) |
                bit
            )

        output.append(
            byte
        )

    return bytes(output)


# ============================================================
# COEFFICIENT EMBEDDING
# ============================================================

def embed_bit(
    coefficient,
    bit
):
    """
    Embed one bit using coefficient parity.

        even coefficient -> 0
        odd coefficient  -> 1

    Only a change of 1 is required.
    """

    coefficient = int(
        coefficient
    )

    bit = int(
        bit
    )

    if (
        coefficient & 1
    ) == bit:

        return coefficient

    if coefficient >= 0:

        return coefficient + 1

    return coefficient - 1


# ============================================================
# COEFFICIENT EXTRACTION
# ============================================================

def extract_bit(
    coefficient
):
    """
    Extract bit from coefficient parity.
    """

    coefficient = int(
        coefficient
    )

    return abs(
        coefficient
    ) & 1


# ============================================================
# GET EMBEDDING COEFFICIENTS
# ============================================================

def get_coefficients(
    LH,
    HL
):
    """
    Return coefficient locations.

    Order:

        1. All LH coefficients
        2. All HL coefficients

    The same order is used during
    embedding and extraction.
    """

    locations = []

    # --------------------------------------------------------
    # LH
    # --------------------------------------------------------

    for row in range(
        LH.shape[0]
    ):

        for col in range(
            LH.shape[1]
        ):

            locations.append(
                (
                    "LH",
                    row,
                    col
                )
            )

    # --------------------------------------------------------
    # HL
    # --------------------------------------------------------

    for row in range(
        HL.shape[0]
    ):

        for col in range(
            HL.shape[1]
        ):

            locations.append(
                (
                    "HL",
                    row,
                    col
                )
            )

    return locations


# ============================================================
# EMBED PAYLOAD
# ============================================================

def embed_payload(
    cover_image_path,
    payload_file_path,
    output_path
):
    """
    Embed any supported payload.

    Supported payload types:

        TEXT
        AUDIO
        IMAGE
    """

    print("\n" + "=" * 70)
    print("DWT PAYLOAD EMBEDDING")
    print("=" * 70)

    # ========================================================
    # LOAD PAYLOAD
    # ========================================================

    payload = create_payload_from_file(
        payload_file_path
    )

    payload_bits = bytes_to_bits(
        payload
    )

    # ========================================================
    # LOAD COVER IMAGE
    # ========================================================

    image = cv2.imread(
        cover_image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load cover image: "
            f"{cover_image_path}"
        )

    # ========================================================
    # APPLY DWT
    # ========================================================

    LL, LH, HL, HH = apply_dwt(
        image
    )

    # ========================================================
    # GET EMBEDDING LOCATIONS
    # ========================================================

    locations = get_coefficients(
        LH,
        HL
    )

    capacity_bits = len(
        locations
    )

    required_bits = len(
        payload_bits
    )

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    print(
        f"Cover image       : "
        f"{cover_image_path}"
    )

    print(
        f"Image size        : "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    print(
        f"Payload file      : "
        f"{payload_file_path}"
    )

    print(
        f"Payload size      : "
        f"{len(payload):,} bytes"
    )

    print(
        f"Required bits     : "
        f"{required_bits:,}"
    )

    print(
        f"Capacity bits     : "
        f"{capacity_bits:,}"
    )

    utilization = (
        required_bits /
        capacity_bits *
        100
    )

    print(
        f"Capacity usage    : "
        f"{utilization:.2f}%"
    )

    # ========================================================
    # CAPACITY CHECK
    # ========================================================

    if required_bits > capacity_bits:

        print(
            "\nRESULT: "
            "PAYLOAD DOES NOT FIT [FAIL]"
        )

        raise ValueError(
            f"DWT capacity insufficient. "
            f"Required {required_bits:,} bits, "
            f"available {capacity_bits:,} bits."
        )

    # ========================================================
    # EMBEDDING
    # ========================================================

    start_time = time.perf_counter()

    for index, bit in enumerate(
        payload_bits
    ):

        band, row, col = (
            locations[index]
        )

        if band == "LH":

            LH[row, col] = embed_bit(
                LH[row, col],
                bit
            )

        else:

            HL[row, col] = embed_bit(
                HL[row, col],
                bit
            )

    # ========================================================
    # INVERSE DWT
    # ========================================================

    reconstructed = apply_idwt(
        LL,
        LH,
        HL,
        HH
    )

    # ========================================================
    # CONVERT TO VALID IMAGE
    # ========================================================

    reconstructed = np.clip(
        reconstructed,
        0,
        255
    ).astype(
        np.uint8
    )

    embedding_time = (
        time.perf_counter() -
        start_time
    )

    # ========================================================
    # SAVE STEGO IMAGE
    # ========================================================

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    success = cv2.imwrite(
        output_path,
        reconstructed
    )

    if not success:

        raise IOError(
            f"Could not save stego image: "
            f"{output_path}"
        )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print(
        f"Bits embedded     : "
        f"{required_bits:,}"
    )

    print(
        f"Embedding time    : "
        f"{embedding_time:.4f} sec"
    )

    print(
        f"Stego image       : "
        f"{output_path}"
    )

    print("=" * 70)

    return {
        "required_bits":
            required_bits,

        "capacity_bits":
            capacity_bits,

        "utilization":
            utilization,

        "embedding_time":
            embedding_time,
    }


# ============================================================
# EXTRACT BITS FROM COEFFICIENTS
# ============================================================

def extract_bits_from_coefficients(
    LH,
    HL,
    number_of_bits
):
    """
    Extract a specified number of bits from
    LH + HL coefficients.
    """

    locations = get_coefficients(
        LH,
        HL
    )

    if number_of_bits > len(
        locations
    ):

        raise ValueError(
            "Requested more bits than "
            "DWT capacity."
        )

    bits = []

    for index in range(
        number_of_bits
    ):

        band, row, col = (
            locations[index]
        )

        if band == "LH":

            coefficient = LH[
                row,
                col
            ]

        else:

            coefficient = HL[
                row,
                col
            ]

        bits.append(
            extract_bit(
                coefficient
            )
        )

    return bits


# ============================================================
# EXTRACT PAYLOAD
# ============================================================

def extract_payload(
    stego_image_path,
    output_directory
):
    """
    Extract the complete AISG payload from
    a DWT stego image.
    """

    print("\n" + "=" * 70)
    print("DWT PAYLOAD EXTRACTION")
    print("=" * 70)

    start_time = time.perf_counter()

    # ========================================================
    # LOAD STEGO IMAGE
    # ========================================================

    image = cv2.imread(
        stego_image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load stego image: "
            f"{stego_image_path}"
        )

    # ========================================================
    # APPLY DWT
    # ========================================================

    _, LH, HL, _ = apply_dwt(
        image
    )

    # ========================================================
    # CAPACITY
    # ========================================================

    capacity_bits = (
        LH.size +
        HL.size
    )

    # ========================================================
    # EXTRACT MINIMUM HEADER
    # ========================================================

    # AISG header:
    #
    # MAGIC          = 4 bytes
    # VERSION        = 1 byte
    # PAYLOAD TYPE   = 1 byte
    # EXTENSION LEN  = 1 byte
    # EXTENSION      = variable
    # PAYLOAD SIZE   = 8 bytes
    #
    # Minimum:
    #
    # 4 + 1 + 1 + 1 + 8 = 15 bytes

    minimum_header_bytes = 15

    minimum_header_bits = (
        minimum_header_bytes * 8
    )

    if minimum_header_bits > capacity_bits:

        raise ValueError(
            "DWT capacity is too small "
            "to contain the AISG header."
        )

    header_bits = (
        extract_bits_from_coefficients(
            LH,
            HL,
            minimum_header_bits
        )
    )

    first_header = bits_to_bytes(
        header_bits
    )

    # ========================================================
    # VERIFY MAGIC
    # ========================================================

    if first_header[:4] != b"AISG":

        raise ValueError(
            "Invalid DWT payload. "
            "MAGIC mismatch. "
            "The embedded header could not "
            "be recovered correctly."
        )

    print(
        f"MAGIC             : "
        f"{first_header[:4].decode(errors='replace')}"
    )

    # ========================================================
    # READ HEADER FIELDS
    # ========================================================

    version = first_header[4]

    payload_type = first_header[5]

    extension_length = first_header[6]

    print(
        f"Version           : "
        f"{version}"
    )

    print(
        f"Extension length  : "
        f"{extension_length} bytes"
    )

    # ========================================================
    # DETERMINE FULL HEADER SIZE
    # ========================================================

    full_header_size = (
        4 +
        1 +
        1 +
        1 +
        extension_length +
        8
    )

    full_header_bits_required = (
        full_header_size * 8
    )

    if (
        full_header_bits_required >
        capacity_bits
    ):

        raise ValueError(
            "Extracted header size exceeds "
            "DWT capacity."
        )

    # ========================================================
    # EXTRACT COMPLETE HEADER
    # ========================================================

    full_header_bits = (
        extract_bits_from_coefficients(
            LH,
            HL,
            full_header_bits_required
        )
    )

    full_header = bits_to_bytes(
        full_header_bits
    )

    # ========================================================
    # READ PAYLOAD SIZE DIRECTLY FROM HEADER
    # ========================================================

    # Header layout:
    #
    # 0 - 3       MAGIC
    # 4           VERSION
    # 5           PAYLOAD TYPE
    # 6           EXTENSION LENGTH
    # 7 ...       EXTENSION
    # next 8      PAYLOAD SIZE
    #
    # Therefore payload size starts at:
    #
    # 7 + extension_length

    payload_size_offset = (
        7 +
        extension_length
    )

    payload_size_end = (
        payload_size_offset +
        8
    )

    if payload_size_end > len(
        full_header
    ):

        raise ValueError(
            "Could not recover the "
            "complete payload size field."
        )

    payload_size_bytes = full_header[
        payload_size_offset:
        payload_size_end
    ]

    # AISG payload size is stored as
    # an unsigned 8-byte integer.
    #
    # This follows the common big-endian
    # binary header representation used
    # by the payload format.

    payload_size = int.from_bytes(
        payload_size_bytes,
        byteorder="big",
        signed=False
    )

    print(
        f"Payload size      : "
        f"{payload_size:,} bytes"
    )

    # ========================================================
    # DETERMINE TOTAL PAYLOAD SIZE
    # ========================================================

    total_payload_size = (
        full_header_size +
        payload_size
    )

    total_bits = (
        total_payload_size * 8
    )

    # ========================================================
    # CAPACITY CHECK
    # ========================================================

    if total_bits > capacity_bits:

        raise ValueError(
            "Extracted payload size exceeds "
            "DWT capacity. "
            "The stego image may be corrupted."
        )

    # ========================================================
    # EXTRACT COMPLETE PAYLOAD
    # ========================================================

    payload_bits = (
        extract_bits_from_coefficients(
            LH,
            HL,
            total_bits
        )
    )

    payload = bits_to_bytes(
        payload_bits
    )

    # ========================================================
    # PARSE COMPLETE PAYLOAD
    # ========================================================

    # IMPORTANT:
    #
    # Do NOT call parse_payload() on
    # full_header alone.
    #
    # parse_payload() expects the complete
    # AISG structure including payload data.

    info = parse_payload(
        payload
    )

    # ========================================================
    # SAVE EXTRACTED DATA
    # ========================================================

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    extension = info[
        "extension"
    ]

    if extension:

        filename = (
            f"dwt_extracted."
            f"{extension}"
        )

    else:

        filename = (
            "dwt_extracted"
        )

    output_path = os.path.join(
        output_directory,
        filename
    )

    with open(
        output_path,
        "wb"
    ) as file:

        file.write(
            info["data"]
        )

    # ========================================================
    # EXTRACTION TIME
    # ========================================================

    extraction_time = (
        time.perf_counter() -
        start_time
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print(
        f"Stego image       : "
        f"{stego_image_path}"
    )

    print(
        f"Image size        : "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    print(
        f"Payload type      : "
        f"{info['payload_type']}"
    )

    print(
        f"Extension         : "
        f".{extension}"
    )

    print(
        f"Payload size      : "
        f"{info['size']:,} bytes"
    )

    print(
        f"Extraction time   : "
        f"{extraction_time:.4f} sec"
    )

    print(
        f"Output file       : "
        f"{output_path}"
    )

    print("=" * 70)

    return {
        "payload_type":
            info["payload_type"],

        "extension":
            extension,

        "size":
            info["size"],

        "output_path":
            output_path,

        "extraction_time":
            extraction_time,
    }


# ============================================================
# TEST
# ============================================================

def run_test():

    cover_image = (
        "data/images/processed/"
        "cover_processed.png"
    )

    test_payload = (
        "data/payloads/"
        "test.txt"
    )

    stego_image = (
        "output/stego/"
        "dwt_test.png"
    )

    extraction_directory = (
        "output/extracted/"
        "dwt_test"
    )

    print("\n" + "=" * 70)
    print("DWT STEGANOGRAPHY TEST")
    print("=" * 70)

    # ========================================================
    # CAPACITY
    # ========================================================

    capacity_bits, capacity_bytes = (
        calculate_capacity(
            cover_image
        )
    )

    print("\nDWT CAPACITY")
    print("-" * 70)

    print(
        f"Capacity : "
        f"{capacity_bits:,} bits"
    )

    print(
        f"Capacity : "
        f"{capacity_bytes:,} bytes"
    )

    # ========================================================
    # EMBEDDING
    # ========================================================

    embed_result = embed_payload(
        cover_image,
        test_payload,
        stego_image
    )

    # ========================================================
    # EXTRACTION
    # ========================================================

    extract_result = extract_payload(
        stego_image,
        extraction_directory
    )

    # ========================================================
    # INTEGRITY CHECK
    # ========================================================

    with open(
        test_payload,
        "rb"
    ) as file:

        original_data = file.read()

    with open(
        extract_result["output_path"],
        "rb"
    ) as file:

        extracted_data = file.read()

    print("\nINTEGRITY CHECK")
    print("-" * 70)

    print(
        f"Original bytes    : "
        f"{len(original_data):,}"
    )

    print(
        f"Extracted bytes   : "
        f"{len(extracted_data):,}"
    )

    if original_data == extracted_data:

        print(
            "Data integrity    : "
            "100% [OK]"
        )

        print(
            "Result             : "
            "PASS [OK]"
        )

    else:

        print(
            "Data integrity    : "
            "FAILED [FAIL]"
        )

        print(
            "Result             : "
            "FAIL [FAIL]"
        )

        raise ValueError(
            "DWT extraction does not "
            "match original payload."
        )

    print("\n" + "=" * 70)
    print("DWT TEST COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_test()