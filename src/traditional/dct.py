"""
DCT-Based Image Steganography

Generic payload support:
    TEXT  -> .txt
    AUDIO -> .wav / .mp3 / etc.
    IMAGE -> .png / .jpg / etc.

Method:
    - Convert cover image to grayscale
    - Divide image into 8x8 blocks
    - Apply 2D DCT
    - Embed one bit using a pair of mid-frequency DCT coefficients
    - Use the common AISG payload format
    - Extract and reconstruct the original payload
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

BLOCK_SIZE = 8

# Two mid-frequency coefficients.
# Using a pair allows us to represent one bit based on
# which coefficient has the larger magnitude.
COEFF_A = (4, 3)
COEFF_B = (3, 4)

# Minimum difference between the two coefficients.
# This helps make the embedded bit more robust.
MIN_DIFFERENCE = 10.0


# ============================================================
# CAPACITY
# ============================================================

def calculate_capacity(image_path):
    """
    Calculate the DCT payload capacity.

    One bit is stored in each 8x8 block.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width = image.shape

    blocks_y = height // BLOCK_SIZE
    blocks_x = width // BLOCK_SIZE

    total_blocks = blocks_y * blocks_x

    capacity_bits = total_blocks
    capacity_bytes = capacity_bits // 8

    return {
        "width": width,
        "height": height,
        "blocks_y": blocks_y,
        "blocks_x": blocks_x,
        "total_blocks": total_blocks,
        "capacity_bits": capacity_bits,
        "capacity_bytes": capacity_bytes,
    }


# ============================================================
# BIT CONVERSION
# ============================================================

def bytes_to_bits(data):
    """Convert bytes to a list of bits."""

    bits = []

    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)

    return bits


def bits_to_bytes(bits):
    """Convert a list of bits back to bytes."""

    if len(bits) % 8 != 0:
        raise ValueError("Number of bits must be divisible by 8.")

    output = bytearray()

    for i in range(0, len(bits), 8):
        byte = 0

        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit

        output.append(byte)

    return bytes(output)


# ============================================================
# DCT COEFFICIENT EMBEDDING
# ============================================================

def embed_bit(block, bit):
    """
    Embed one bit into an 8x8 DCT block.

    bit = 0:
        coefficient A > coefficient B

    bit = 1:
        coefficient B > coefficient A
    """

    dct_block = cv2.dct(np.float32(block - 128.0))

    a_row, a_col = COEFF_A
    b_row, b_col = COEFF_B

    a = dct_block[a_row, a_col]
    b = dct_block[b_row, b_col]

    # Make sure the required ordering exists.
    if bit == 0:

        if a <= b + MIN_DIFFERENCE:
            center = (a + b) / 2.0

            dct_block[a_row, a_col] = center + MIN_DIFFERENCE / 2
            dct_block[b_row, b_col] = center - MIN_DIFFERENCE / 2

    else:

        if b <= a + MIN_DIFFERENCE:
            center = (a + b) / 2.0

            dct_block[a_row, a_col] = center - MIN_DIFFERENCE / 2
            dct_block[b_row, b_col] = center + MIN_DIFFERENCE / 2

    # Inverse DCT
    reconstructed = cv2.idct(dct_block) + 128.0

    reconstructed = np.clip(
        np.round(reconstructed),
        0,
        255
    ).astype(np.uint8)

    return reconstructed


# ============================================================
# DCT BIT EXTRACTION
# ============================================================

def extract_bit(block):
    """
    Extract one bit from an 8x8 DCT block.
    """

    dct_block = cv2.dct(
        np.float32(block.astype(np.float32) - 128.0)
    )

    a_row, a_col = COEFF_A
    b_row, b_col = COEFF_B

    a = dct_block[a_row, a_col]
    b = dct_block[b_row, b_col]

    if a > b:
        return 0

    return 1


# ============================================================
# EMBEDDING
# ============================================================

def embed_payload(
    cover_image_path,
    payload_file_path,
    output_path
):
    """
    Embed any supported payload into a cover image using DCT.
    """

    print("\n" + "=" * 70)
    print("DCT PAYLOAD EMBEDDING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load cover image
    # --------------------------------------------------------

    image = cv2.imread(
        cover_image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Cover image not found: {cover_image_path}"
        )

    height, width = image.shape

    print(f"Cover image       : {cover_image_path}")
    print(f"Image size        : {width} x {height}")

    # --------------------------------------------------------
    # Create common payload
    # --------------------------------------------------------

    payload = create_payload_from_file(
        payload_file_path
    )

    payload_size = len(payload)

    print(f"Payload file      : {payload_file_path}")
    print(f"Payload size      : {payload_size:,} bytes")

    # --------------------------------------------------------
    # Convert payload to bits
    # --------------------------------------------------------

    bits = bytes_to_bits(payload)

    required_bits = len(bits)

    capacity = calculate_capacity(
        cover_image_path
    )

    capacity_bits = capacity["capacity_bits"]

    print(f"Required bits     : {required_bits:,}")
    print(f"Capacity bits     : {capacity_bits:,}")

    if required_bits > capacity_bits:

        print("\nRESULT: PAYLOAD DOES NOT FIT [FAIL]")

        raise ValueError(
            f"DCT capacity insufficient. "
            f"Required {required_bits:,} bits, "
            f"available {capacity_bits:,} bits."
        )

    utilization = (
        required_bits / capacity_bits
    ) * 100

    print(
        f"Capacity usage    : {utilization:.2f}%"
    )

    # --------------------------------------------------------
    # Start embedding
    # --------------------------------------------------------

    stego = image.copy()

    bit_index = 0

    start_time = time.perf_counter()

    blocks_y = height // BLOCK_SIZE
    blocks_x = width // BLOCK_SIZE

    for by in range(blocks_y):

        for bx in range(blocks_x):

            if bit_index >= required_bits:
                break

            y = by * BLOCK_SIZE
            x = bx * BLOCK_SIZE

            block = stego[
                y:y + BLOCK_SIZE,
                x:x + BLOCK_SIZE
            ]

            stego[
                y:y + BLOCK_SIZE,
                x:x + BLOCK_SIZE
            ] = embed_bit(
                block,
                bits[bit_index]
            )

            bit_index += 1

        if bit_index >= required_bits:
            break

    embedding_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    success = cv2.imwrite(
        output_path,
        stego
    )

    if not success:
        raise IOError(
            f"Could not save stego image: {output_path}"
        )

    print(f"Bits embedded     : {bit_index:,}")
    print(f"Embedding time    : {embedding_time:.4f} sec")
    print(f"Stego image       : {output_path}")

    print("=" * 70)

    return {
        "payload_size": payload_size,
        "required_bits": required_bits,
        "capacity_bits": capacity_bits,
        "utilization_percent": utilization,
        "embedding_time": embedding_time,
    }


# ============================================================
# EXTRACTION
# ============================================================

def extract_payload(
    stego_image_path,
    output_directory
):
    """
    Extract the common AISG payload from a DCT stego image.
    """

    print("\n" + "=" * 70)
    print("DCT PAYLOAD EXTRACTION")
    print("=" * 70)

    image = cv2.imread(
        stego_image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Stego image not found: {stego_image_path}"
        )

    height, width = image.shape

    print(f"Stego image       : {stego_image_path}")
    print(f"Image size        : {width} x {height}")

    start_time = time.perf_counter()

    bits = []

    blocks_y = height // BLOCK_SIZE
    blocks_x = width // BLOCK_SIZE

    # --------------------------------------------------------
    # We first need enough bits to reconstruct the header.
    #
    # The common header has:
    #
    # MAGIC       = 4 bytes
    # VERSION     = 1 byte
    # TYPE        = 1 byte
    # EXT LENGTH  = 1 byte
    #
    # We first extract 7 bytes.
    # --------------------------------------------------------

    minimum_header_bytes = 7
    minimum_header_bits = minimum_header_bytes * 8

    for by in range(blocks_y):

        for bx in range(blocks_x):

            if len(bits) >= minimum_header_bits:
                break

            y = by * BLOCK_SIZE
            x = bx * BLOCK_SIZE

            block = image[
                y:y + BLOCK_SIZE,
                x:x + BLOCK_SIZE
            ]

            bits.append(
                extract_bit(block)
            )

        if len(bits) >= minimum_header_bits:
            break

    header_prefix = bits_to_bytes(bits)

    if header_prefix[:4] != b"AISG":
        raise ValueError(
            "Invalid DCT stego image: "
            "AISG MAGIC not found."
        )

    version = header_prefix[4]

    if version != 1:
        raise ValueError(
            f"Unsupported payload version: {version}"
        )

    extension_length = header_prefix[6]

    # --------------------------------------------------------
    # Header size:
    #
    # 4 MAGIC
    # 1 VERSION
    # 1 TYPE
    # 1 EXTENSION LENGTH
    # extension
    # 8 PAYLOAD SIZE
    # --------------------------------------------------------

    complete_header_size = (
        4
        + 1
        + 1
        + 1
        + extension_length
        + 8
    )

    complete_header_bits = (
        complete_header_size * 8
    )

    # --------------------------------------------------------
    # Extract complete header
    # --------------------------------------------------------

    bits = []

    total_blocks = blocks_y * blocks_x

    for block_number in range(total_blocks):

        by = block_number // blocks_x
        bx = block_number % blocks_x

        y = by * BLOCK_SIZE
        x = bx * BLOCK_SIZE

        block = image[
            y:y + BLOCK_SIZE,
            x:x + BLOCK_SIZE
        ]

        bits.append(
            extract_bit(block)
        )

        if len(bits) >= complete_header_bits:
            break

    header_bytes = bits_to_bytes(bits)

    # --------------------------------------------------------
    # Read payload size
    # --------------------------------------------------------

    try:
        extension_start = 7

        extension_end = (
            extension_start
            + extension_length
        )

        payload_size_start = extension_end
        payload_size_end = (
            payload_size_start + 8
        )

        payload_size = int.from_bytes(
            header_bytes[
                payload_size_start:
                payload_size_end
            ],
            byteorder="big"
        )

    except Exception as error:

        raise ValueError(
            "Could not decode payload header."
        ) from error

    # --------------------------------------------------------
    # Total payload bits
    # --------------------------------------------------------

    total_payload_size = (
        complete_header_size
        + payload_size
    )

    total_payload_bits = (
        total_payload_size * 8
    )

    if total_payload_bits > total_blocks:
        raise ValueError(
            "Encoded payload size exceeds "
            "DCT image capacity."
        )

    # --------------------------------------------------------
    # Extract complete payload
    # --------------------------------------------------------

    bits = []

    for block_number in range(total_payload_bits):

        by = block_number // blocks_x
        bx = block_number % blocks_x

        y = by * BLOCK_SIZE
        x = bx * BLOCK_SIZE

        block = image[
            y:y + BLOCK_SIZE,
            x:x + BLOCK_SIZE
        ]

        bits.append(
            extract_bit(block)
        )

    extracted_payload = bits_to_bytes(bits)

    # --------------------------------------------------------
    # Parse common payload
    # --------------------------------------------------------

    payload_info = parse_payload(
        extracted_payload
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    extension = payload_info["extension"]

    output_file = os.path.join(
        output_directory,
        f"dct_extracted.{extension}"
    )

    with open(
        output_file,
        "wb"
    ) as file:
        file.write(
            payload_info["data"]
        )

    extraction_time = (
        time.perf_counter() - start_time
    )

    print(f"Payload type      : {payload_info['payload_type']}")
    print(f"Extension         : .{extension}")
    print(
        f"Payload size      : "
        f"{payload_info['size']:,} bytes"
    )
    print(
        f"Extraction time   : "
        f"{extraction_time:.4f} sec"
    )
    print(f"Output file       : {output_file}")

    print("=" * 70)

    return {
        "payload_type": payload_info["payload_type"],
        "extension": extension,
        "size": payload_info["size"],
        "data": payload_info["data"],
        "output_path": output_file,
        "extraction_time": extraction_time,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DCT STEGANOGRAPHY TEST")
    print("=" * 70)

    cover_image = (
        "data/images/processed/"
        "cover_processed.png"
    )

    test_payload = (
        "data/payloads/test.txt"
    )

    stego_image = (
        "output/stego/dct_test.png"
    )

    extraction_directory = (
        "output/extracted/dct_test"
    )

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    print("\nDCT CAPACITY")
    print("-" * 70)

    capacity = calculate_capacity(
        cover_image
    )

    print(
        f"Image size        : "
        f"{capacity['width']} x "
        f"{capacity['height']}"
    )

    print(
        f"8x8 blocks        : "
        f"{capacity['total_blocks']:,}"
    )

    print(
        f"Capacity          : "
        f"{capacity['capacity_bits']:,} bits"
    )

    print(
        f"Capacity          : "
        f"{capacity['capacity_bytes']:,} bytes"
    )

    # --------------------------------------------------------
    # Embed
    # --------------------------------------------------------

    embed_result = embed_payload(
        cover_image,
        test_payload,
        stego_image
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    extract_result = extract_payload(
        stego_image,
        extraction_directory
    )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    with open(
        test_payload,
        "rb"
    ) as file:
        original_data = file.read()

    extracted_data = extract_result["data"]

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
            "Data integrity    : 100% [OK]"
        )

        print(
            "Result             : PASS [OK]"
        )

    else:

        print(
            "Data integrity    : FAILED [FAIL]"
        )

        print(
            "Result             : FAIL [FAIL]"
        )

        raise AssertionError(
            "Original and extracted payload "
            "are not identical."
        )

    print("\n" + "=" * 70)
    print("DCT TEST COMPLETE")
    print("=" * 70)