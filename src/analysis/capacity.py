from pathlib import Path
from PIL import Image
import wave
import math


# ============================================================
# PATHS
# ============================================================

IMAGE_PATH = Path("data/images/processed/cover_processed.png")
AUDIO_PATH = Path("data/audio/processed/audio_processed.wav")


# ============================================================
# CONSTANTS
# ============================================================

# DCT uses 8x8 blocks
DCT_BLOCK_SIZE = 8

# Number of coefficients used per block
# We will initially use 2 bits per 8x8 block
DCT_BITS_PER_BLOCK = 2

# DWT configuration
# Initial theoretical capacity based on LH + HL subbands
DWT_SUBBANDS = 2


# ============================================================
# AUDIO INFORMATION
# ============================================================

def get_audio_info(audio_path):
    with wave.open(str(audio_path), "rb") as audio:

        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frames = audio.getnframes()

        payload_bytes = frames * channels * sample_width
        duration = frames / sample_rate

    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration": duration,
        "payload_bytes": payload_bytes,
    }


# ============================================================
# IMAGE INFORMATION
# ============================================================

def get_image_info(image_path):

    image = Image.open(image_path)

    width, height = image.size
    mode = image.mode

    # RGB = 3 channels
    if mode == "RGB":
        channels = 3
    elif mode == "RGBA":
        channels = 4
    elif mode == "L":
        channels = 1
    else:
        channels = len(image.getbands())

    total_pixels = width * height

    return {
        "width": width,
        "height": height,
        "mode": mode,
        "channels": channels,
        "total_pixels": total_pixels,
    }


# ============================================================
# LSB CAPACITY
# ============================================================

def calculate_lsb_capacity(image_info):

    pixels = image_info["total_pixels"]
    channels = image_info["channels"]

    # 1 LSB per channel
    capacity_bits = pixels * channels
    capacity_bytes = capacity_bits // 8

    return capacity_bits, capacity_bytes


# ============================================================
# DCT CAPACITY
# ============================================================

def calculate_dct_capacity(image_info):

    width = image_info["width"]
    height = image_info["height"]

    blocks_x = width // DCT_BLOCK_SIZE
    blocks_y = height // DCT_BLOCK_SIZE

    total_blocks = blocks_x * blocks_y

    capacity_bits = total_blocks * DCT_BITS_PER_BLOCK
    capacity_bytes = capacity_bits // 8

    return {
        "blocks_x": blocks_x,
        "blocks_y": blocks_y,
        "total_blocks": total_blocks,
        "capacity_bits": capacity_bits,
        "capacity_bytes": capacity_bytes,
    }


# ============================================================
# DWT CAPACITY
# ============================================================

def calculate_dwt_capacity(image_info):

    width = image_info["width"]
    height = image_info["height"]

    # One-level Haar DWT:
    # Each subband is approximately half width x half height
    subband_width = width // 2
    subband_height = height // 2

    coefficients_per_subband = subband_width * subband_height

    # Using LH and HL subbands
    total_coefficients = coefficients_per_subband * DWT_SUBBANDS

    # 1 bit per coefficient
    capacity_bits = total_coefficients
    capacity_bytes = capacity_bits // 8

    return {
        "subband_width": subband_width,
        "subband_height": subband_height,
        "coefficients_per_subband": coefficients_per_subband,
        "total_coefficients": total_coefficients,
        "capacity_bits": capacity_bits,
        "capacity_bytes": capacity_bytes,
    }


# ============================================================
# FORMAT BYTES
# ============================================================

def format_bytes(size):

    if size < 1024:
        return f"{size:.2f} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"

    if size < 1024 ** 3:
        return f"{size / (1024 ** 2):.2f} MB"

    return f"{size / (1024 ** 3):.2f} GB"


# ============================================================
# CHECK CAPACITY
# ============================================================

def check_capacity(method, capacity_bytes, payload_bytes):

    if capacity_bytes >= payload_bytes:

        utilization = (payload_bytes / capacity_bytes) * 100

        return (
            f"FIT ✅ | "
            f"Utilization: {utilization:.2f}%"
        )

    shortage = payload_bytes - capacity_bytes

    return (
        f"DOES NOT FIT ❌ | "
        f"Shortage: {format_bytes(shortage)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CAPACITY ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not IMAGE_PATH.exists():

        print(f"\nERROR: Image not found:")
        print(IMAGE_PATH)
        return

    if not AUDIO_PATH.exists():

        print(f"\nERROR: Audio not found:")
        print(AUDIO_PATH)
        return

    # --------------------------------------------------------
    # Read information
    # --------------------------------------------------------

    image_info = get_image_info(IMAGE_PATH)
    audio_info = get_audio_info(AUDIO_PATH)

    # --------------------------------------------------------
    # Display image information
    # --------------------------------------------------------

    print("\nIMAGE INFORMATION")
    print("-" * 60)

    print(f"Image file       : {IMAGE_PATH}")
    print(
        f"Image size       : "
        f"{image_info['width']} x {image_info['height']}"
    )
    print(f"Image mode       : {image_info['mode']}")
    print(f"Channels         : {image_info['channels']}")
    print(f"Total pixels     : {image_info['total_pixels']:,}")

    # --------------------------------------------------------
    # Display audio information
    # --------------------------------------------------------

    print("\nAUDIO PAYLOAD INFORMATION")
    print("-" * 60)

    print(f"Audio file       : {AUDIO_PATH}")
    print(f"Sample rate      : {audio_info['sample_rate']} Hz")
    print(f"Channels         : {audio_info['channels']}")
    print(f"Sample width     : {audio_info['sample_width']} bytes")
    print(f"Frames           : {audio_info['frames']:,}")
    print(f"Duration         : {audio_info['duration']:.2f} seconds")

    payload_bytes = audio_info["payload_bytes"]
    payload_bits = payload_bytes * 8

    print(f"Audio payload    : {format_bytes(payload_bytes)}")
    print(f"Payload bits     : {payload_bits:,}")

    # --------------------------------------------------------
    # LSB
    # --------------------------------------------------------

    lsb_bits, lsb_bytes = calculate_lsb_capacity(image_info)

    print("\n" + "=" * 60)
    print("1. LSB CAPACITY")
    print("=" * 60)

    print(f"Bits per channel : 1")
    print(f"Capacity         : {lsb_bits:,} bits")
    print(f"Capacity         : {format_bytes(lsb_bytes)}")
    print(
        f"Result           : "
        f"{check_capacity('LSB', lsb_bytes, payload_bytes)}"
    )

    # --------------------------------------------------------
    # DCT
    # --------------------------------------------------------

    dct = calculate_dct_capacity(image_info)

    print("\n" + "=" * 60)
    print("2. DCT CAPACITY")
    print("=" * 60)

    print(f"Block size       : {DCT_BLOCK_SIZE} x {DCT_BLOCK_SIZE}")
    print(f"Blocks X         : {dct['blocks_x']}")
    print(f"Blocks Y         : {dct['blocks_y']}")
    print(f"Total blocks     : {dct['total_blocks']:,}")
    print(f"Bits per block   : {DCT_BITS_PER_BLOCK}")
    print(f"Capacity         : {dct['capacity_bits']:,} bits")
    print(f"Capacity         : {format_bytes(dct['capacity_bytes'])}")
    print(
        f"Result           : "
        f"{check_capacity('DCT', dct['capacity_bytes'], payload_bytes)}"
    )

    # --------------------------------------------------------
    # DWT
    # --------------------------------------------------------

    dwt = calculate_dwt_capacity(image_info)

    print("\n" + "=" * 60)
    print("3. DWT CAPACITY")
    print("=" * 60)

    print(f"Wavelet          : Haar")
    print(f"Subbands used    : LH + HL")
    print(
        f"Subband size     : "
        f"{dwt['subband_width']} x {dwt['subband_height']}"
    )
    print(
        f"Coefficients/sub : "
        f"{dwt['coefficients_per_subband']:,}"
    )
    print(
        f"Total coefficients: "
        f"{dwt['total_coefficients']:,}"
    )
    print(f"Bits/coeff       : 1")
    print(f"Capacity         : {dwt['capacity_bits']:,} bits")
    print(f"Capacity         : {format_bytes(dwt['capacity_bytes'])}")
    print(
        f"Result           : "
        f"{check_capacity('DWT', dwt['capacity_bytes'], payload_bytes)}"
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CAPACITY COMPARISON")
    print("=" * 60)

    print(
        f"{'Method':<10}"
        f"{'Capacity':<18}"
        f"{'Required':<18}"
        f"{'Status'}"
    )

    print("-" * 60)

    print(
        f"{'LSB':<10}"
        f"{format_bytes(lsb_bytes):<18}"
        f"{format_bytes(payload_bytes):<18}"
        f"{'FIT' if lsb_bytes >= payload_bytes else 'NO'}"
    )

    print(
        f"{'DCT':<10}"
        f"{format_bytes(dct['capacity_bytes']):<18}"
        f"{format_bytes(payload_bytes):<18}"
        f"{'FIT' if dct['capacity_bytes'] >= payload_bytes else 'NO'}"
    )

    print(
        f"{'DWT':<10}"
        f"{format_bytes(dwt['capacity_bytes']):<18}"
        f"{format_bytes(payload_bytes):<18}"
        f"{'FIT' if dwt['capacity_bytes'] >= payload_bytes else 'NO'}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()