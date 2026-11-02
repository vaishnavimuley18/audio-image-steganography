# Multi-Modal Audio-Image Steganography with AI Method Recommendation

A comparative research platform and end-to-end framework for hiding multi-modal payloads (**Audio**, **Image**, and **Text**) inside high-resolution digital images using spatial, frequency, and wavelet steganography combined with machine learning method recommendation.

---

## 🌟 Key Features

- **Multi-Modal Payload Support**: Embed and extract arbitrary **Audio (.wav)**, **Image (.png, .jpg)**, and **Text (.txt)** files.
- **Unified Binary Protocol (`AISG`)**: Custom serialization protocol encoding magic bytes, format version, payload type, original file extension, and exact length to guarantee 100% byte-accurate payload extraction.
- **Three Steganography Engines**:
  - **LSB (Least Significant Bit)**: High-capacity spatial domain embedding capable of hiding large audio files (~6 MB capacity in 4000×4000 images).
  - **DCT (Discrete Cosine Transform)**: Frequency domain embedding across 8×8 blocks using mid-frequency coefficient pairs with differential thresholding for high compression resilience.
  - **DWT (Discrete Wavelet Transform)**: Wavelet domain embedding via reversible Integer Haar Transform targeting LH and HL detail subbands for superior visual imperceptibility.
- **AI Recommendation Engine**: Trained `RandomForestRegressor` evaluating cover image texture, Shannon entropy, Canny edge density, and payload properties to predict and recommend the optimal method with highest multi-objective utility.
- **Dual User Interfaces**:
  - **Streamlit Web Application** (`web_app.py`): In-browser embedding, extraction, audio player, image preview, and metric visualizers.
  - **Tkinter Desktop GUI** (`src/gui/app.py`): Native desktop graphical interface with styled cards and live processing.
- **Quantitative Quality Metrics**: Real-time computation of Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Mean Squared Error (MSE), and Changed Pixel Percentage.

---

## 📊 Steganography Algorithms & Performance

| Method | Domain | Embedding Strategy | Capacity (4000×4000) | Avg PSNR | Avg SSIM | Best Use Case |
|---|---|---|---|---|---|---|
| **LSB** | Spatial | 1 bit / RGB channel | **6,000,000 Bytes** (~6.0 MB) | **68.49 dB** | **0.9983** | High-capacity payloads (full audio files, large images) |
| **DCT** | Frequency | 1 bit / 8×8 block (mid-freq pairs) | **31,250 Bytes** (~31.2 KB) | **59.90 dB** | **0.9969** | Small text payloads, robust against compression |
| **DWT** | Wavelet | Integer Haar (LH & HL subbands) | **1,000,000 Bytes** (~1.0 MB) | **70.34 dB** | **0.9992** | High perceptual quality, medium payloads |

---

## 📦 Project Structure

```
audio-image-steganography/
├── main.py                     # Unified entry point (Web, GUI, CLI, Benchmark)
├── web_app.py                  # Streamlit web dashboard
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── audio/                  # Raw input WAV files
│   ├── images/
│   │   └── covers/             # 10 diverse cover images (cover_01 - cover_10)
│   └── payloads/
│       ├── test.txt, test.wav, test.png
│       └── variants/           # Size variants (small, medium, large for audio, image, text)
│
├── src/
│   ├── utils/
│   │   └── payload_format.py   # AISG binary header protocol
│   ├── preprocessing/
│   │   ├── audio_processor.py  # Audio resampling (44.1kHz mono PCM-16) & normalization
│   │   ├── image_processor.py  # Aspect-ratio-preserving 4000×4000 canvas preprocessing
│   │   └── payload_processor.py# Payload validation & extraction
│   ├── traditional/
│   │   ├── lsb.py              # Spatial LSB steganography engine
│   │   ├── dct.py              # Frequency DCT 8×8 steganography engine
│   │   └── dwt.py              # Integer Haar Wavelet steganography engine
│   ├── analysis/
│   │   ├── capacity.py         # Theoretical capacity calculations
│   │   ├── metrics.py          # PSNR, SSIM, MSE, changed pixel calculation
│   │   ├── image_features.py   # Cover image feature extraction (entropy, edge density)
│   │   ├── payload_features.py # Payload metadata & dimension extraction
│   │   ├── recommender_model.py# Inference engine for ML recommender
│   │   ├── train_recommender.py# Random Forest training pipeline
│   │   ├── auto_recommend.py   # Cover & payload evaluation pipeline
│   │   └── experiment_runner.py# Benchmark runner across covers, methods, and payloads
│   └── gui/
│       └── app.py              # Native desktop Tkinter interface
│
└── results/                    # Multi-cover benchmark data, plots, and trained ML model
    ├── experimental_results.csv
    ├── multi_cover_results.csv
    ├── comparison/             # Performance comparison plots (PSNR, SSIM, execution time)
    └── multi_cover_analysis/   # Recommender model (recommender_model.joblib) & diagnostics
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install requirements in a virtual environment:

```bash
git clone https://github.com/vaishnavimuley18/audio-image-steganography.git
cd audio-image-steganography

python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Web Application (Streamlit)

```bash
python main.py --web
# or
streamlit run web_app.py --server.port 8501
```
Open **http://localhost:8501** in your browser to access:
- **🔒 Hide Data Tab**: Choose/upload cover and payload, request AI recommendation, embed data, and inspect metrics.
- **🔓 Extract Data Tab**: Recover payload from stego images with in-browser audio playback, image preview, or text viewer.

### 3. Launch Native Desktop GUI (Tkinter)

```bash
python main.py
# or
python src/gui/app.py
```

### 4. Run Benchmark Experiments via CLI

```bash
python main.py --experiments
```

### 5. Run Preprocessing Pipeline via CLI

```bash
python main.py --cli
```

---

## 🧪 Experimental Findings

From systematic testing across 10 cover images and 9 payload variations (90 experiments per method):
- **LSB**: Achieved **100% success rate** and **100% byte integrity** across all payload types including 1 MB audio files, with an average PSNR of **68.49 dB** and SSIM of **0.9983**.
- **DWT**: Showed the highest visual quality (**70.34 dB avg PSNR**, **0.9992 avg SSIM**) for payloads fitting within its 1 MB capacity.
- **DCT**: Best suited for compact text payloads and applications requiring JPEG compression resistance.
- **Recommender Model**: Accurately avoids capacity failures and selects the technique yielding the highest multi-objective utility score ($R^2 > 0.98$).

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
