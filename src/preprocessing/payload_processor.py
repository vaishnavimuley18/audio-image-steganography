"""
Generic Payload Processor

Converts TEXT, AUDIO, and IMAGE files into the
common steganography payload format.

Supported payloads:
    TEXT  -> .txt
    AUDIO -> .wav, .mp3, .flac, etc.
    IMAGE -> .png, .jpg, .jpeg, .bmp, etc.
"""

from pathlib import Path

from src.utils.payload_format import (
    create_payload,
    parse_payload,
    print_payload_info,
)


# ============================================================
# SUPPORTED EXTENSIONS
# ============================================================

TEXT_EXTENSIONS = {
    ".txt"
}

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a"
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp"
}


# ============================================================
# DETECT PAYLOAD TYPE
# ============================================================

def detect_payload_type(file_path):
    """
    Detect payload type from file extension.

    Parameters
    ----------
    file_path : str or Path
        Path to payload file

    Returns
    -------
    str
        TEXT, AUDIO, or IMAGE
    """

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension in TEXT_EXTENSIONS:
        return "TEXT"

    if extension in AUDIO_EXTENSIONS:
        return "AUDIO"

    if extension in IMAGE_EXTENSIONS:
        return "IMAGE"

    raise ValueError(
        f"Unsupported payload file extension: {extension}"
    )


# ============================================================
# READ PAYLOAD FILE
# ============================================================

def read_payload_file(file_path):
    """
    Read payload file as raw bytes.

    Parameters
    ----------
    file_path : str or Path
        Path to payload file

    Returns
    -------
    bytes
        Raw file data
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Payload file not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    return file_path.read_bytes()


# ============================================================
# CREATE FORMATTED PAYLOAD FROM FILE
# ============================================================

def create_payload_from_file(file_path):
    """
    Convert an actual payload file into the
    common steganography payload format.

    Parameters
    ----------
    file_path : str or Path
        Path to payload file

    Returns
    -------
    bytes
        Complete formatted payload
    """

    file_path = Path(file_path)

    payload_type = detect_payload_type(file_path)

    data = read_payload_file(file_path)

    extension = file_path.suffix.lower().lstrip(".")

    payload = create_payload(
        payload_type=payload_type,
        extension=extension,
        data=data
    )

    return payload


# ============================================================
# EXTRACT FILE DATA FROM FORMATTED PAYLOAD
# ============================================================

def extract_payload_data(payload, output_directory):
    """
    Extract the original payload data from the
    common payload format.

    Parameters
    ----------
    payload : bytes
        Complete formatted payload

    output_directory : str or Path
        Directory where extracted file will be saved

    Returns
    -------
    Path
        Path of extracted file
    """

    info = parse_payload(payload)

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    extension = info["extension"]

    output_file = (
        output_directory
        / f"extracted_payload.{extension}"
    )

    output_file.write_bytes(info["data"])

    return output_file


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENERIC PAYLOAD PROCESSOR TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Test files
    # --------------------------------------------------------

    test_files = [
        "data/payloads/test.txt",
        "data/payloads/test.wav",
        "data/payloads/test.png",
    ]

    for file_path in test_files:

        print("\n" + "-" * 60)
        print(f"Testing: {file_path}")
        print("-" * 60)

        try:

            # Create formatted payload
            payload = create_payload_from_file(file_path)

            # Display information
            print_payload_info(payload)

            # Parse payload
            info = parse_payload(payload)

            # Verify
            original_data = Path(file_path).read_bytes()

            assert info["data"] == original_data

            print(f"Detected type : {info['payload_type']}")
            print(f"Original size : {len(original_data):,} bytes")
            print(f"Payload size  : {info['size']:,} bytes")
            print("Data integrity: 100%")
            print("Result        : PASS [OK]")

        except Exception as error:

            print(f"Result        : FAIL [FAIL]")
            print(f"Error         : {error}")

    print("\n" + "=" * 60)
    print("GENERIC PAYLOAD PROCESSOR TEST COMPLETE")
    print("=" * 60)