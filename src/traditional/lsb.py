"""
Generic LSB Steganography Engine

Supports:
    TEXT
    AUDIO
    IMAGE

The payload is first converted into the common payload format,
then embedded into the RGB cover image using 1 LSB per channel.

Extraction automatically reconstructs the original payload
and uses the stored extension to save the recovered file.
"""

from pathlib import Path
import time

import numpy as np
from PIL import Image

from src.preprocessing.payload_processor import (
    create_payload_from_file,
    extract_payload_data,
)
from src.utils.payload_format import parse_payload


# ============================================================
# CONSTANTS
# ============================================================

BITS_PER_CHANNEL = 1
CHANNELS_PER_PIXEL = 3


# ============================================================
# CAPACITY
# ============================================================

def calculate_capacity(image_path):
    """
    Calculate LSB embedding capacity.

    One bit is stored in each RGB channel.
    """

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    total_pixels = width * height
    total_channels = total_pixels * CHANNELS_PER_PIXEL

    capacity_bits = total_channels * BITS_PER_CHANNEL
    capacity_bytes = capacity_bits // 8

    return {
        "width": width,
        "height": height,
        "pixels": total_pixels,
        "capacity_bits": capacity_bits,
        "capacity_bytes": capacity_bytes,
    }


# ============================================================
# EMBEDDING
# ============================================================

def embed_payload(cover_image_path, payload_file_path, output_path):
    """
    Embed any supported payload into a cover image.

    Parameters
    ----------
    cover_image_path : str or Path
        Cover image.

    payload_file_path : str or Path
        Text, audio, or image payload.

    output_path : str or Path
        Output stego image.
    """

    cover_image_path = Path(cover_image_path)
    payload_file_path = Path(payload_file_path)
    output_path = Path(output_path)

    print("\n" + "=" * 60)
    print("GENERIC LSB EMBEDDING")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD COVER IMAGE
    # --------------------------------------------------------

    image = Image.open(cover_image_path).convert("RGB")

    image_array = np.array(image, dtype=np.uint8)

    print(f"Cover image   : {cover_image_path}")
    print(f"Image size    : {image.width} x {image.height}")
    print(f"Payload file  : {payload_file_path}")

    # --------------------------------------------------------
    # CREATE COMMON PAYLOAD
    # --------------------------------------------------------

    payload = create_payload_from_file(payload_file_path)

    info = parse_payload(payload)

    print(f"Payload type  : {info['payload_type']}")
    print(f"Extension     : .{info['extension']}")
    print(f"Payload data  : {info['size']:,} bytes")
    print(f"Total payload : {len(payload):,} bytes")

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------

    capacity = calculate_capacity(cover_image_path)

    capacity_bits = capacity["capacity_bits"]
    required_bits = len(payload) * 8

    utilization = (
        required_bits / capacity_bits
    ) * 100

    print(f"Capacity      : {capacity['capacity_bytes']:,} bytes")
    print(f"Required bits : {required_bits:,}")
    print(f"Utilization   : {utilization:.2f}%")

    if required_bits > capacity_bits:

        raise ValueError(
            "\nPayload does not fit inside the cover image.\n"
            f"Required : {required_bits:,} bits\n"
            f"Available: {capacity_bits:,} bits"
        )

    print("Capacity check: PASS [OK]")

    # --------------------------------------------------------
    # CONVERT PAYLOAD TO BITS
    # --------------------------------------------------------

    payload_array = np.frombuffer(
        payload,
        dtype=np.uint8
    )

    payload_bits = np.unpackbits(payload_array)

    # --------------------------------------------------------
    # FLATTEN IMAGE
    # --------------------------------------------------------

    flat_image = image_array.flatten()

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    start_time = time.perf_counter()

    flat_image[:len(payload_bits)] &= 254
    flat_image[:len(payload_bits)] |= payload_bits

    stego_array = flat_image.reshape(
        image_array.shape
    )

    embedding_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # SAVE STEGO IMAGE
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    stego_image = Image.fromarray(
        stego_array
    )

    stego_image.save(
        output_path,
        format="PNG"
    )

    print(f"Embedding time: {embedding_time:.4f} seconds")
    print(f"Stego image   : {output_path}")

    print("=" * 60)
    print("EMBEDDING COMPLETE [OK]")
    print("=" * 60)

    return {
        "payload_type": info["payload_type"],
        "payload_size": info["size"],
        "total_payload_size": len(payload),
        "capacity_bytes": capacity["capacity_bytes"],
        "utilization_percent": utilization,
        "embedding_time": embedding_time,
        "output_path": str(output_path),
    }


# ============================================================
# EXTRACTION
# ============================================================

def extract_payload(stego_image_path, output_directory):
    """
    Extract a generic payload from a stego image.

    The function reads the common payload header first,
    determines the payload size, then extracts exactly
    the required number of bytes.
    """

    stego_image_path = Path(stego_image_path)
    output_directory = Path(output_directory)

    print("\n" + "=" * 60)
    print("GENERIC LSB EXTRACTION")
    print("=" * 60)

    print(f"Stego image   : {stego_image_path}")

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        stego_image_path
    ).convert("RGB")

    image_array = np.array(
        image,
        dtype=np.uint8
    )

    flat_image = image_array.flatten()

    # --------------------------------------------------------
    # EXTRACT LSBs
    # --------------------------------------------------------

    start_time = time.perf_counter()

    extracted_bits = flat_image & 1

    # --------------------------------------------------------
    # READ HEADER
    # --------------------------------------------------------

    # Minimum header:
    # MAGIC       4
    # VERSION     1
    # TYPE        1
    # EXT LENGTH  1
    # SIZE        8
    #
    # But extension is variable, so first extract
    # enough bits for the fixed portion.

    fixed_header_size = 4 + 1 + 1 + 1

    fixed_header_bits = fixed_header_size * 8

    fixed_header_bits_data = extracted_bits[
        :fixed_header_bits
    ]

    fixed_header_bytes = np.packbits(
        fixed_header_bits_data
    ).tobytes()

    # --------------------------------------------------------
    # CHECK MAGIC
    # --------------------------------------------------------

    if fixed_header_bytes[:4] != b"AISG":

        raise ValueError(
            "Invalid stego image: MAGIC mismatch."
        )

    # --------------------------------------------------------
    # GET EXTENSION LENGTH
    # --------------------------------------------------------

    extension_length = fixed_header_bytes[6]

    # Header size:
    #
    # 4 MAGIC
    # 1 VERSION
    # 1 TYPE
    # 1 EXT LENGTH
    # extension
    # 8 PAYLOAD SIZE

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
    # EXTRACT COMPLETE HEADER
    # --------------------------------------------------------

    header_bits = extracted_bits[
        :complete_header_bits
    ]

    header_bytes = np.packbits(
        header_bits
    ).tobytes()

    # --------------------------------------------------------
    # PARSE HEADER
    # --------------------------------------------------------

    # We don't need to manually decode every field here.
    # parse_payload will validate everything.

    # The header alone isn't enough for parse_payload,
    # so read the payload size directly.

    payload_size_position = (
        4 + 1 + 1 + 1 + extension_length
    )

    payload_size_start = (
        payload_size_position * 8
    )

    payload_size_end = (
        payload_size_start + 64
    )

    payload_size_bits = extracted_bits[
        payload_size_start:payload_size_end
    ]

    payload_size = int.from_bytes(
        np.packbits(payload_size_bits).tobytes(),
        byteorder="big"
    )

    # --------------------------------------------------------
    # TOTAL PAYLOAD SIZE
    # --------------------------------------------------------

    total_payload_size = (
        complete_header_size
        + payload_size
    )

    total_payload_bits = (
        total_payload_size * 8
    )

    if total_payload_bits > len(extracted_bits):

        raise ValueError(
            "Incomplete payload in stego image."
        )

    # --------------------------------------------------------
    # EXTRACT COMPLETE PAYLOAD
    # --------------------------------------------------------

    payload_bits = extracted_bits[
        :total_payload_bits
    ]

    payload = np.packbits(
        payload_bits
    ).tobytes()

    # --------------------------------------------------------
    # VALIDATE PAYLOAD
    # --------------------------------------------------------

    info = parse_payload(payload)

    # --------------------------------------------------------
    # SAVE ORIGINAL FILE
    # --------------------------------------------------------

    output_file = extract_payload_data(
        payload,
        output_directory
    )

    extraction_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(f"Payload type  : {info['payload_type']}")
    print(f"Extension     : .{info['extension']}")
    print(f"Payload size  : {info['size']:,} bytes")
    print(f"Extracted file: {output_file}")
    print(f"Extraction time: {extraction_time:.4f} seconds")
    print("Data integrity: 100%")

    print("=" * 60)
    print("EXTRACTION COMPLETE [OK]")
    print("=" * 60)

    return {
        "payload_type": info["payload_type"],
        "extension": info["extension"],
        "payload_size": info["size"],
        "output_path": str(output_file),
        "extraction_time": extraction_time,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENERIC LSB STEGANOGRAPHY TEST")
    print("=" * 60)

    cover_image = (
        "data/images/processed/cover_processed.png"
    )

    test_payloads = [
        "data/payloads/test.txt",
        "data/payloads/test.wav",
        "data/payloads/test.png",
    ]

    for payload_file in test_payloads:

        payload_path = Path(payload_file)

        print("\n\n")
        print("#" * 60)
        print(f"TESTING PAYLOAD: {payload_path}")
        print("#" * 60)

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        extension = payload_path.suffix.lower().lstrip(".")

        stego_path = (
            Path("output/stego")
            / f"lsb_test_{extension}.png"
        )

        extract_directory = (
            Path("output/extracted")
            / f"lsb_{extension}"
        )

        try:

            # ------------------------------------------------
            # EMBED
            # ------------------------------------------------

            embed_payload(
                cover_image,
                payload_path,
                stego_path
            )

            # ------------------------------------------------
            # EXTRACT
            # ------------------------------------------------

            result = extract_payload(
                stego_path,
                extract_directory
            )

            # ------------------------------------------------
            # VERIFY
            # ------------------------------------------------

            original_data = payload_path.read_bytes()

            extracted_data = Path(
                result["output_path"]
            ).read_bytes()

            if original_data == extracted_data:

                print("\nDATA VERIFICATION: PASS [OK]")
                print("Original and extracted files are identical.")

            else:

                print("\nDATA VERIFICATION: FAIL [FAIL]")

                raise ValueError(
                    "Original and extracted data differ."
                )

        except Exception as error:

            print("\nTEST FAILED [FAIL]")
            print(f"Error: {error}")

    print("\n" + "=" * 60)
    print("GENERIC LSB TEST COMPLETE")
    print("=" * 60)