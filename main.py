import sys
import argparse

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_preprocessing():
    from src.preprocessing.audio_processor import preprocess_audio
    from src.preprocessing.image_processor import preprocess_image

    print("Running audio preprocessing...")
    preprocess_audio(
        input_path="data/audio/input.wav",
        output_path="data/audio/processed/audio_processed.wav"
    )

    print("\nRunning image preprocessing...")
    preprocess_image()


def run_experiments():
    from src.analysis.experiment_runner import main as run_benchmark
    run_benchmark()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Audio-Image Steganography"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Run web application on localhost (Streamlit)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run audio & image preprocessing in CLI mode instead of GUI",
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run traditional steganography benchmark experiment suite",
    )

    args = parser.parse_args()

    if args.web:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "web_app.py", "--server.port", "8501"])
    elif args.experiments:
        run_experiments()
    elif args.cli:
        run_preprocessing()
    else:
        from src.gui.app import main as run_gui
        run_gui()


if __name__ == "__main__":
    main()