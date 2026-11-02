import soundfile as sf
import numpy as np
from pathlib import Path


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

INPUT_PATH = Path("data/audio/input.wav")
OUTPUT_PATH = Path("data/audio/processed/audio_processed.wav")

TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 1
MAX_DURATION = 30.0


def preprocess_audio(input_path=None, output_path=None):

    print("=" * 55)
    print("AUDIO PREPROCESSING")
    print("=" * 55)

    input_file = Path(input_path) if input_path else INPUT_PATH
    output_file = Path(output_path) if output_path else OUTPUT_PATH

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input audio not found: {input_file}"
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------
    audio, sample_rate = sf.read(
        input_file,
        always_2d=True
    )

    original_channels = audio.shape[1]

    # Convert to float32 for processing
    audio = audio.astype(np.float32)

    # --------------------------------------------------------
    # Calculate original duration
    # --------------------------------------------------------
    original_frames = len(audio)
    original_duration = original_frames / sample_rate

    print(f"Input file       : {INPUT_PATH}")
    print(f"Format           : WAV")
    print(f"Original subtype : PCM_16")
    print(f"Sample rate      : {sample_rate} Hz")
    print(f"Channels         : {original_channels}")
    print(f"Duration         : {original_duration:.2f} seconds")
    print(f"Frames           : {original_frames}")

    # --------------------------------------------------------
    # Convert to mono
    # --------------------------------------------------------
    if original_channels > 1:
        audio = np.mean(audio, axis=1, keepdims=True)
        print(
            f"Channel conversion: "
            f"{original_channels} -> 1"
        )
    else:
        print("Channel conversion: Already mono")

    # --------------------------------------------------------
    # Resample to 44100 Hz
    # --------------------------------------------------------
    if sample_rate != TARGET_SAMPLE_RATE:

        new_length = int(
            len(audio)
            * TARGET_SAMPLE_RATE
            / sample_rate
        )

        old_indices = np.linspace(
            0,
            1,
            len(audio)
        )

        new_indices = np.linspace(
            0,
            1,
            new_length
        )

        resampled = np.interp(
            new_indices,
            old_indices,
            audio[:, 0]
        )

        audio = resampled.reshape(-1, 1)

        print(
            f"Sample-rate conversion: "
            f"{sample_rate} -> {TARGET_SAMPLE_RATE} Hz"
        )

    else:
        print(
            "Sample-rate conversion: "
            "Not required"
        )

    # --------------------------------------------------------
    # Trim to maximum 30 seconds
    # --------------------------------------------------------
    max_frames = int(
        TARGET_SAMPLE_RATE * MAX_DURATION
    )

    if len(audio) > max_frames:

        audio = audio[:max_frames]

        print(
            f"Duration          : "
            f"Trimmed to {MAX_DURATION:.2f} seconds"
        )

    else:

        print(
            f"Duration          : "
            f"No trimming required"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save as PCM-16 WAV
    # --------------------------------------------------------
    sf.write(
        output_file,
        audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16"
    )

    # --------------------------------------------------------
    # Read output information
    # --------------------------------------------------------
    processed_audio, processed_sr = sf.read(
        output_file,
        always_2d=True
    )

    processed_channels = processed_audio.shape[1]
    processed_frames = len(processed_audio)

    processed_duration = (
        processed_frames / processed_sr
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------
    print(f"Output file       : {output_file}")
    print(f"Output format     : WAV")
    print(f"Output subtype    : PCM_16")
    print(f"Output sample rate: {processed_sr} Hz")
    print(f"Output channels   : {processed_channels}")
    print(f"Output duration   : {processed_duration:.2f} seconds")

    print()
    print("Audio preprocessing completed successfully! [OK]")

    return {
        "output_path": output_file,
        "sample_rate": processed_sr,
        "channels": processed_channels,
        "duration": processed_duration,
        "frames": processed_frames,
    }


if __name__ == "__main__":
    preprocess_audio()