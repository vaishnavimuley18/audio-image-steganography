"""
Common Payload Format for Multi-Modal Image Steganography

Supported payload types:
    TEXT  -> 01
    AUDIO -> 02
    IMAGE -> 03

Payload structure:

+----------------+----------+
| Field          | Size     |
+----------------+----------+
| Magic          | 4 bytes  |
| Version        | 1 byte   |
| Payload Type   | 1 byte   |
| Extension Len  | 1 byte   |
| Extension      | Variable |
| Payload Size   | 8 bytes  |
| Payload Data   | Variable |
+----------------+----------+
"""

import struct


# ============================================================
# CONSTANTS
# ============================================================

MAGIC = b"AISG"
VERSION = 1

PAYLOAD_TYPES = {
    "TEXT": 1,
    "AUDIO": 2,
    "IMAGE": 3,
}

PAYLOAD_TYPE_NAMES = {
    1: "TEXT",
    2: "AUDIO",
    3: "IMAGE",
}


# ============================================================
# CREATE PAYLOAD
# ============================================================

def create_payload(payload_type, extension, data):
    """
    Create a complete steganography payload.

    Parameters
    ----------
    payload_type : str
        TEXT, AUDIO, or IMAGE

    extension : str
        File extension such as txt, wav, png, jpg

    data : bytes
        Actual payload data

    Returns
    -------
    bytes
        Complete formatted payload
    """

    payload_type = payload_type.upper()

    if payload_type not in PAYLOAD_TYPES:
        raise ValueError(
            f"Unsupported payload type: {payload_type}"
        )

    if not isinstance(data, bytes):
        raise TypeError("Payload data must be bytes.")

    extension = extension.lower().lstrip(".")

    extension_bytes = extension.encode("utf-8")

    if len(extension_bytes) > 255:
        raise ValueError("File extension is too long.")

    payload_size = len(data)

    # Header:
    # MAGIC          -> 4 bytes
    # VERSION        -> 1 byte
    # PAYLOAD TYPE   -> 1 byte
    # EXTENSION LEN  -> 1 byte
    # EXTENSION      -> variable
    # PAYLOAD SIZE   -> 8 bytes

    header = (
        MAGIC
        + struct.pack("B", VERSION)
        + struct.pack("B", PAYLOAD_TYPES[payload_type])
        + struct.pack("B", len(extension_bytes))
        + extension_bytes
        + struct.pack(">Q", payload_size)
    )

    return header + data


# ============================================================
# PARSE PAYLOAD
# ============================================================

def parse_payload(payload):
    """
    Parse a complete steganography payload.

    Parameters
    ----------
    payload : bytes
        Complete formatted payload

    Returns
    -------
    dict
        Information about the payload and actual data
    """

    if not isinstance(payload, bytes):
        raise TypeError("Payload must be bytes.")

    minimum_header_size = 4 + 1 + 1 + 1 + 8

    if len(payload) < minimum_header_size:
        raise ValueError("Payload is too small.")

    position = 0

    # --------------------------------------------------------
    # MAGIC
    # --------------------------------------------------------

    magic = payload[position:position + 4]
    position += 4

    if magic != MAGIC:
        raise ValueError("Invalid payload: MAGIC mismatch.")

    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    version = payload[position]
    position += 1

    if version != VERSION:
        raise ValueError(
            f"Unsupported payload version: {version}"
        )

    # --------------------------------------------------------
    # PAYLOAD TYPE
    # --------------------------------------------------------

    payload_type_id = payload[position]
    position += 1

    if payload_type_id not in PAYLOAD_TYPE_NAMES:
        raise ValueError(
            f"Unknown payload type ID: {payload_type_id}"
        )

    payload_type = PAYLOAD_TYPE_NAMES[payload_type_id]

    # --------------------------------------------------------
    # EXTENSION
    # --------------------------------------------------------

    extension_length = payload[position]
    position += 1

    extension_bytes = payload[
        position:position + extension_length
    ]

    position += extension_length

    extension = extension_bytes.decode("utf-8")

    # --------------------------------------------------------
    # PAYLOAD SIZE
    # --------------------------------------------------------

    if position + 8 > len(payload):
        raise ValueError("Incomplete payload header.")

    payload_size = struct.unpack(
        ">Q",
        payload[position:position + 8]
    )[0]

    position += 8

    # --------------------------------------------------------
    # PAYLOAD DATA
    # --------------------------------------------------------

    if position + payload_size > len(payload):
        raise ValueError(
            "Payload data is incomplete or corrupted."
        )

    data = payload[
        position:position + payload_size
    ]

    return {
        "version": version,
        "payload_type": payload_type,
        "extension": extension,
        "size": payload_size,
        "data": data,
    }


# ============================================================
# DISPLAY PAYLOAD INFORMATION
# ============================================================

def print_payload_info(payload):
    """
    Print human-readable payload information.
    """

    info = parse_payload(payload)

    print("\n" + "=" * 60)
    print("PAYLOAD INFORMATION")
    print("=" * 60)

    print(f"Version       : {info['version']}")
    print(f"Payload type  : {info['payload_type']}")
    print(f"Extension     : .{info['extension']}")
    print(f"Payload size  : {info['size']:,} bytes")
    print(f"Total size    : {len(payload):,} bytes")

    print("=" * 60)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("COMMON PAYLOAD FORMAT TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # TEST 1: TEXT
    # --------------------------------------------------------

    print("\nTEST 1: TEXT")
    print("-" * 60)

    text = "Hello! This is a steganography test."

    text_data = text.encode("utf-8")

    text_payload = create_payload(
        "TEXT",
        "txt",
        text_data
    )

    print_payload_info(text_payload)

    parsed_text = parse_payload(text_payload)

    recovered_text = parsed_text["data"].decode("utf-8")

    print(f"Original text  : {text}")
    print(f"Recovered text : {recovered_text}")

    assert text == recovered_text

    print("Result         : PASS [OK]")

    # --------------------------------------------------------
    # TEST 2: AUDIO
    # --------------------------------------------------------

    print("\nTEST 2: AUDIO")
    print("-" * 60)

    audio_data = bytes(range(256)) * 100

    audio_payload = create_payload(
        "AUDIO",
        "wav",
        audio_data
    )

    print_payload_info(audio_payload)

    parsed_audio = parse_payload(audio_payload)

    assert parsed_audio["data"] == audio_data
    assert parsed_audio["payload_type"] == "AUDIO"
    assert parsed_audio["extension"] == "wav"

    print("Result         : PASS [OK]")

    # --------------------------------------------------------
    # TEST 3: IMAGE
    # --------------------------------------------------------

    print("\nTEST 3: IMAGE")
    print("-" * 60)

    image_data = bytes(range(256)) * 200

    image_payload = create_payload(
        "IMAGE",
        "png",
        image_data
    )

    print_payload_info(image_payload)

    parsed_image = parse_payload(image_payload)

    assert parsed_image["data"] == image_data
    assert parsed_image["payload_type"] == "IMAGE"
    assert parsed_image["extension"] == "png"

    print("Result         : PASS [OK]")

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COMMON PAYLOAD FORMAT TEST COMPLETE [OK]")
    print("=" * 60)