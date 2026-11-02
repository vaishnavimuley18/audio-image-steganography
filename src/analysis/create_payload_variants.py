from pathlib import Path
import wave
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAYLOAD_DIR = PROJECT_ROOT / "data" / "payloads"
VARIANT_DIR = PAYLOAD_DIR / "variants"

TEXT_SOURCE = PAYLOAD_DIR / "test.txt"
AUDIO_SOURCE = PAYLOAD_DIR / "test.wav"
IMAGE_SOURCE = PAYLOAD_DIR / "test.png"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

VARIANT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT VARIANTS
# ============================================================

def create_text_variants():
    print("\n" + "=" * 60)
    print("CREATING TEXT PAYLOAD VARIANTS")
    print("=" * 60)

    if not TEXT_SOURCE.exists():
        raise FileNotFoundError(f"Text source not found: {TEXT_SOURCE}")

    original_text = TEXT_SOURCE.read_text(encoding="utf-8")

    # Target sizes in bytes
    targets = {
        "small": 1 * 1024,       # 1 KB
        "medium": 10 * 1024,     # 10 KB
        "large": 50 * 1024,      # 50 KB
    }

    for name, target_size in targets.items():

        repeated_text = (original_text + "\n") * (
            target_size // max(len(original_text) + 1, 1) + 2
        )

        # Trim to approximately target size
        text_bytes = repeated_text.encode("utf-8")
        text_bytes = text_bytes[:target_size]

        output_path = VARIANT_DIR / f"text_{name}.txt"

        output_path.write_bytes(text_bytes)

        print(
            f"{name.capitalize():8} : "
            f"{output_path.stat().st_size:,} bytes "
            f"({output_path.stat().st_size / 1024:.2f} KB)"
        )


# ============================================================
# AUDIO VARIANTS
# ============================================================

def create_audio_variants():
    print("\n" + "=" * 60)
    print("CREATING AUDIO PAYLOAD VARIANTS")
    print("=" * 60)

    if not AUDIO_SOURCE.exists():
        raise FileNotFoundError(f"Audio source not found: {AUDIO_SOURCE}")

    durations = {
        "small": 2,
        "medium": 5,
        "large": 10,
    }

    with wave.open(str(AUDIO_SOURCE), "rb") as source:

        params = source.getparams()

        frame_rate = source.getframerate()
        total_frames = source.getnframes()

        print(f"Source sample rate : {frame_rate} Hz")
        print(f"Source channels    : {params.nchannels}")
        print(f"Source width       : {params.sampwidth} bytes")
        print(f"Source duration    : {total_frames / frame_rate:.2f} sec")

        for name, duration in durations.items():

            required_frames = int(frame_rate * duration)

            # Never request more frames than the source contains
            frames_to_read = min(required_frames, total_frames)

            source.setpos(0)
            audio_frames = source.readframes(frames_to_read)

            output_path = VARIANT_DIR / f"audio_{name}.wav"

            with wave.open(str(output_path), "wb") as output:

                output.setnchannels(params.nchannels)
                output.setsampwidth(params.sampwidth)
                output.setframerate(frame_rate)
                output.writeframes(audio_frames)

            actual_duration = frames_to_read / frame_rate
            file_size = output_path.stat().st_size

            print(
                f"{name.capitalize():8} : "
                f"{file_size:,} bytes "
                f"({file_size / 1024:.2f} KB), "
                f"{actual_duration:.2f} sec"
            )


# ============================================================
# IMAGE VARIANTS
# ============================================================

def create_image_variants():
    print("\n" + "=" * 60)
    print("CREATING IMAGE PAYLOAD VARIANTS")
    print("=" * 60)

    if not IMAGE_SOURCE.exists():
        raise FileNotFoundError(f"Image source not found: {IMAGE_SOURCE}")

    with Image.open(IMAGE_SOURCE) as image:

        print(f"Source size : {image.width} x {image.height}")
        print(f"Source mode : {image.mode}")

        # Different resolutions create different payload sizes
        sizes = {
            "small": (256, 256),
            "medium": (512, 512),
            "large": (900, 1600),
        }

        # Convert to RGB for consistency
        image = image.convert("RGB")

        for name, size in sizes.items():

            resized = image.resize(
                size,
                Image.Resampling.LANCZOS
            )

            output_path = VARIANT_DIR / f"image_{name}.png"

            resized.save(
                output_path,
                format="PNG"
            )

            file_size = output_path.stat().st_size

            print(
                f"{name.capitalize():8} : "
                f"{size[0]} x {size[1]}, "
                f"{file_size:,} bytes "
                f"({file_size / 1024:.2f} KB)"
            )


# ============================================================
# VERIFY ALL VARIANTS
# ============================================================

def verify_variants():
    print("\n" + "=" * 60)
    print("VERIFYING PAYLOAD VARIANTS")
    print("=" * 60)

    expected_files = [
        "text_small.txt",
        "text_medium.txt",
        "text_large.txt",

        "audio_small.wav",
        "audio_medium.wav",
        "audio_large.wav",

        "image_small.png",
        "image_medium.png",
        "image_large.png",
    ]

    all_valid = True

    for filename in expected_files:

        path = VARIANT_DIR / filename

        if path.exists() and path.stat().st_size > 0:

            print(
                f"✓ {filename:20} "
                f"{path.stat().st_size:,} bytes"
            )

        else:

            print(f"✗ {filename:20} MISSING")

            all_valid = False

    print()

    if all_valid:
        print("✓ ALL 9 PAYLOAD VARIANTS CREATED SUCCESSFULLY!")
    else:
        print("✗ SOME PAYLOAD VARIANTS ARE MISSING.")

    return all_valid


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("PAYLOAD VARIANT GENERATOR")
    print("=" * 60)

    create_text_variants()
    create_audio_variants()
    create_image_variants()

    success = verify_variants()

    print("\n" + "=" * 60)

    if success:
        print("PAYLOAD VARIANT GENERATION COMPLETE ✓")
    else:
        print("PAYLOAD VARIANT GENERATION FAILED ✗")

    print("=" * 60)


if __name__ == "__main__":
    main()