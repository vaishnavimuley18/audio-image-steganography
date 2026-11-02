import sys
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# Ensure project root is in sys.path when running app.py directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from src.traditional import lsb
from src.traditional import dct
from src.traditional import dwt
from src.analysis.auto_recommend import auto_recommend


# ============================================================
# APPLICATION
# ============================================================

class SteganographyApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Multi-Modal Image Steganography"
        )

        self.root.geometry(
            "900x650"
        )

        self.root.minsize(
            800,
            600
        )

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        self.cover_path = None
        self.payload_path = None

        self.last_stego_path = None
        self.last_method = None
        self.last_metrics = None

        self.method_var = tk.StringVar(
            value="AUTO"
        )

        self.payload_type_var = tk.StringVar(
            value="Not selected"
        )

        self.recommendation_var = tk.StringVar(
            value="No recommendation yet"
        )

        # ----------------------------------------------------
        # Styling
        # ----------------------------------------------------

        self.bg = "#F7F4EF"
        self.card_bg = "#FFFFFF"
        self.text = "#252525"
        self.muted = "#6B6B6B"
        self.accent = "#7B1E3A"
        self.border = "#DDD6CE"

        self.root.configure(
            bg=self.bg
        )

        self.build_ui()


    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = tk.Frame(
            self.root,
            bg=self.accent,
            height=90
        )

        header.pack(
            fill="x"
        )

        title = tk.Label(
            header,
            text="Multi-Modal Image Steganography",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg=self.accent
        )

        title.pack(
            pady=(18, 2)
        )

        subtitle = tk.Label(
            header,
            text="LSB • DCT • DWT • AI-Based Method Recommendation",
            font=("Segoe UI", 10),
            fg="#F3E8EC",
            bg=self.accent
        )

        subtitle.pack()


        # ----------------------------------------------------
        # Main container
        # ----------------------------------------------------

        main = tk.Frame(
            self.root,
            bg=self.bg
        )

        main.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=25
        )


        # ----------------------------------------------------
        # Cover section
        # ----------------------------------------------------

        cover_card = self.create_card(
            main
        )

        cover_card.pack(
            fill="x",
            pady=(0, 15)
        )

        tk.Label(
            cover_card,
            text="1. Cover Image",
            font=("Segoe UI", 14, "bold"),
            fg=self.text,
            bg=self.card_bg
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 8)
        )

        cover_row = tk.Frame(
            cover_card,
            bg=self.card_bg
        )

        cover_row.pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        self.cover_label = tk.Label(
            cover_row,
            text="No cover image selected",
            font=("Segoe UI", 10),
            fg=self.muted,
            bg=self.card_bg
        )

        self.cover_label.pack(
            side="left",
            fill="x",
            expand=True
        )

        tk.Button(
            cover_row,
            text="Select Image",
            command=self.select_cover,
            font=("Segoe UI", 10, "bold"),
            bg=self.accent,
            fg="white",
            relief="flat",
            padx=18,
            pady=7,
            cursor="hand2"
        ).pack(
            side="right"
        )


        # ----------------------------------------------------
        # Payload section
        # ----------------------------------------------------

        payload_card = self.create_card(
            main
        )

        payload_card.pack(
            fill="x",
            pady=(0, 15)
        )

        tk.Label(
            payload_card,
            text="2. Payload",
            font=("Segoe UI", 14, "bold"),
            fg=self.text,
            bg=self.card_bg
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 8)
        )

        payload_row = tk.Frame(
            payload_card,
            bg=self.card_bg
        )

        payload_row.pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        self.payload_label = tk.Label(
            payload_row,
            text="No payload selected",
            font=("Segoe UI", 10),
            fg=self.muted,
            bg=self.card_bg
        )

        self.payload_label.pack(
            side="left",
            fill="x",
            expand=True
        )

        tk.Button(
            payload_row,
            text="Select Payload",
            command=self.select_payload,
            font=("Segoe UI", 10, "bold"),
            bg=self.accent,
            fg="white",
            relief="flat",
            padx=18,
            pady=7,
            cursor="hand2"
        ).pack(
            side="right"
        )

        type_row = tk.Frame(
            payload_card,
            bg=self.card_bg
        )

        type_row.pack(
            fill="x",
            padx=20,
            pady=(0, 15)
        )

        tk.Label(
            type_row,
            text="Detected payload type:",
            font=("Segoe UI", 10),
            fg=self.text,
            bg=self.card_bg
        ).pack(
            side="left"
        )

        tk.Label(
            type_row,
            textvariable=self.payload_type_var,
            font=("Segoe UI", 10, "bold"),
            fg=self.accent,
            bg=self.card_bg
        ).pack(
            side="left",
            padx=8
        )


        # ----------------------------------------------------
        # Method section
        # ----------------------------------------------------

        method_card = self.create_card(
            main
        )

        method_card.pack(
            fill="x",
            pady=(0, 15)
        )

        tk.Label(
            method_card,
            text="3. Steganography Method",
            font=("Segoe UI", 14, "bold"),
            fg=self.text,
            bg=self.card_bg
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 10)
        )

        method_row = tk.Frame(
            method_card,
            bg=self.card_bg
        )

        method_row.pack(
            padx=20,
            pady=(0, 15)
        )

        for method in [
            "AUTO",
            "LSB",
            "DCT",
            "DWT"
        ]:

            tk.Radiobutton(
                method_row,
                text=method,
                variable=self.method_var,
                value=method,
                font=("Segoe UI", 10, "bold"),
                bg=self.card_bg,
                fg=self.text,
                activebackground=self.card_bg,
                selectcolor=self.card_bg
            ).pack(
                side="left",
                padx=(0, 25)
            )


        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        recommendation_card = self.create_card(
            main
        )

        recommendation_card.pack(
            fill="x",
            pady=(0, 15)
        )

        tk.Label(
            recommendation_card,
            text="AI Recommendation",
            font=("Segoe UI", 14, "bold"),
            fg=self.text,
            bg=self.card_bg
        ).pack(
            anchor="w",
            padx=20,
            pady=(15, 5)
        )

        tk.Label(
            recommendation_card,
            textvariable=self.recommendation_var,
            font=("Segoe UI", 12, "bold"),
            fg=self.accent,
            bg=self.card_bg
        ).pack(
            anchor="w",
            padx=20,
            pady=(0, 15)
        )


        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_row = tk.Frame(
            main,
            bg=self.bg
        )

        button_row.pack(
            fill="x",
            pady=5
        )

        tk.Button(
            button_row,
            text="Get AI Recommendation",
            command=self.get_recommendation,
            font=("Segoe UI", 11, "bold"),
            bg=self.accent,
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        ).pack(
            side="left"
        )

        tk.Button(
            button_row,
            text="Hide Data",
            command=self.hide_data_placeholder,
            font=("Segoe UI", 11, "bold"),
            bg="#4A4A4A",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        ).pack(
            side="left",
            padx=10
        )

        tk.Button(
            button_row,
            text="Extract Data",
            command=self.extract_data_placeholder,
            font=("Segoe UI", 11, "bold"),
            bg="#4A4A4A",
            fg="white",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        ).pack(
            side="left"
        )


    # ========================================================
    # CARD
    # ========================================================

    def create_card(self, parent):

        return tk.Frame(
            parent,
            bg=self.card_bg,
            highlightbackground=self.border,
            highlightthickness=1
        )


    # ========================================================
    # SELECT COVER
    # ========================================================

    def select_cover(self):

        path = filedialog.askopenfilename(
            title="Select Cover Image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png"
                )
            ]
        )

        if not path:
            return

        self.cover_path = Path(path)

        self.cover_label.config(
            text=self.cover_path.name,
            fg=self.text
        )

        self.recommendation_var.set(
            "No recommendation yet"
        )

        self.last_stego_path = None
        self.last_method = None
        self.last_metrics = None


    # ========================================================
    # SELECT PAYLOAD
    # ========================================================

    def select_payload(self):

        path = filedialog.askopenfilename(
            title="Select Payload",
            filetypes=[
                (
                    "Supported files",
                    "*.txt *.wav *.png *.jpg *.jpeg"
                )
            ]
        )

        if not path:
            return

        self.payload_path = Path(path)

        self.payload_label.config(
            text=self.payload_path.name,
            fg=self.text
        )

        extension = (
            self.payload_path.suffix.lower()
        )

        if extension == ".txt":

            payload_type = "TEXT"

        elif extension == ".wav":

            payload_type = "AUDIO"

        elif extension in [
            ".png",
            ".jpg",
            ".jpeg"
        ]:

            payload_type = "IMAGE"

        else:

            payload_type = "UNKNOWN"

        self.payload_type_var.set(
            payload_type
        )

        self.recommendation_var.set(
            "No recommendation yet"
        )

        self.last_stego_path = None
        self.last_method = None
        self.last_metrics = None


    # ========================================================
    # AI RECOMMENDATION
    # ========================================================

    def get_recommendation(self):

        if self.cover_path is None:

            messagebox.showwarning(
                "Missing Cover Image",
                "Please select a cover image first."
            )

            return

        if self.payload_path is None:

            messagebox.showwarning(
                "Missing Payload",
                "Please select a payload file first."
            )

            return

        try:

            result = auto_recommend(
                self.cover_path,
                self.payload_path
            )

            method = result[
                "recommended_method"
            ]

            if method is None:

                self.recommendation_var.set(
                    "No feasible method"
                )

                messagebox.showwarning(
                    "No Feasible Method",
                    result.get(
                        "reason",
                        "No suitable method was found."
                    )
                )

                return

            utility = result[
                "predicted_utility"
            ]

            self.method_var.set(
                method
            )

            self.recommendation_var.set(
                f"Recommended Method: {method}    "
                f"Predicted Utility: {utility:.4f}"
            )

        except Exception as error:

            messagebox.showerror(
                "Recommendation Error",
                str(error)
            )


    # ========================================================
    # PREPROCESS COVER
    # ========================================================

    def preprocess_cover(self):

        """
        Converts the selected cover image to a
        4000 x 4000 RGB PNG while preserving aspect ratio.
        """

        output_dir = Path(
            "output/gui"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        processed_path = (
            output_dir /
            "processed_cover.png"
        )

        image = Image.open(
            self.cover_path
        ).convert(
            "RGB"
        )

        target_size = 4000

        original_width, original_height = (
            image.size
        )

        scale = min(
            target_size / original_width,
            target_size / original_height
        )

        new_width = int(
            original_width * scale
        )

        new_height = int(
            original_height * scale
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (target_size, target_size),
            (0, 0, 0)
        )

        x_offset = (
            target_size - new_width
        ) // 2

        y_offset = (
            target_size - new_height
        ) // 2

        canvas.paste(
            image,
            (x_offset, y_offset)
        )

        canvas.save(
            processed_path
        )

        return processed_path


    # ========================================================
    # CALCULATE IMAGE METRICS
    # ========================================================

    def calculate_metrics(
        self,
        cover_image,
        stego_image
    ):

        # ----------------------------------------------------
        # Make sure images are comparable
        # ----------------------------------------------------

        if cover_image is None:

            raise ValueError(
                "Unable to read cover image."
            )

        if stego_image is None:

            raise ValueError(
                "Unable to read stego image."
            )


        # ----------------------------------------------------
        # Convert to grayscale when channel counts differ
        # ----------------------------------------------------

        if (
            len(cover_image.shape) !=
            len(stego_image.shape)
        ):

            cover_gray = cv2.cvtColor(
                cover_image,
                cv2.COLOR_BGR2GRAY
            )

            if len(stego_image.shape) == 3:

                stego_gray = cv2.cvtColor(
                    stego_image,
                    cv2.COLOR_BGR2GRAY
                )

            else:

                stego_gray = stego_image

            cover_image = cover_gray
            stego_image = stego_gray

        elif (
            len(cover_image.shape) == 3
            and
            len(stego_image.shape) == 3
            and
            cover_image.shape[2] != stego_image.shape[2]
        ):

            cover_image = cv2.cvtColor(
                cover_image,
                cv2.COLOR_BGR2GRAY
            )

            stego_image = cv2.cvtColor(
                stego_image,
                cv2.COLOR_BGR2GRAY
            )


        # ----------------------------------------------------
        # Ensure same dimensions
        # ----------------------------------------------------

        if (
            cover_image.shape !=
            stego_image.shape
        ):

            raise ValueError(
                "Cover and stego image dimensions do not match."
            )


        # ----------------------------------------------------
        # Convert to float
        # ----------------------------------------------------

        cover_float = (
            cover_image.astype(
                np.float64
            )
        )

        stego_float = (
            stego_image.astype(
                np.float64
            )
        )


        # ----------------------------------------------------
        # Difference
        # ----------------------------------------------------

        difference = (
            cover_float -
            stego_float
        )


        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        mse = np.mean(
            difference ** 2
        )


        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        if mse == 0:

            psnr = float("inf")

        else:

            psnr = 10 * np.log10(
                (255 ** 2) / mse
            )


        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        if len(cover_image.shape) == 2:

            ssim = structural_similarity(
                cover_image,
                stego_image,
                data_range=255
            )

        else:

            ssim = structural_similarity(
                cover_image,
                stego_image,
                channel_axis=2,
                data_range=255
            )


        # ----------------------------------------------------
        # Changed pixels
        # ----------------------------------------------------

        changed_mask = (
            np.any(
                cover_image != stego_image,
                axis=2
            )
            if len(cover_image.shape) == 3
            else
            cover_image != stego_image
        )

        changed_pixels = np.sum(
            changed_mask
        )

        total_pixels = changed_mask.size

        changed_percentage = (
            changed_pixels /
            total_pixels *
            100
        )


        # ----------------------------------------------------
        # Changed channels
        # ----------------------------------------------------

        if len(cover_image.shape) == 3:

            changed_channels = np.sum(
                cover_image != stego_image
            )

            total_channels = (
                cover_image.shape[0] *
                cover_image.shape[1] *
                cover_image.shape[2]
            )

        else:

            changed_channels = changed_pixels

            total_channels = total_pixels


        changed_channel_percentage = (
            changed_channels /
            total_channels *
            100
        )


        return {

            "mse": float(mse),

            "psnr": float(psnr),

            "ssim": float(ssim),

            "changed_pixels": int(
                changed_pixels
            ),

            "total_pixels": int(
                total_pixels
            ),

            "changed_percentage": float(
                changed_percentage
            ),

            "changed_channels": int(
                changed_channels
            ),

            "total_channels": int(
                total_channels
            ),

            "changed_channel_percentage": float(
                changed_channel_percentage
            )
        }


    # ========================================================
    # HIDE DATA
    # ========================================================

    def hide_data_placeholder(self):

        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        if self.cover_path is None:

            messagebox.showwarning(
                "Missing Cover Image",
                "Please select a cover image first."
            )

            return

        if self.payload_path is None:

            messagebox.showwarning(
                "Missing Payload",
                "Please select a payload file first."
            )

            return


        # ----------------------------------------------------
        # Determine method
        # ----------------------------------------------------

        method = self.method_var.get()


        # ----------------------------------------------------
        # AUTO recommendation
        # ----------------------------------------------------

        if method == "AUTO":

            try:

                recommendation = auto_recommend(
                    self.cover_path,
                    self.payload_path
                )

                method = recommendation[
                    "recommended_method"
                ]

                if method is None:

                    messagebox.showerror(
                        "No Feasible Method",
                        recommendation.get(
                            "reason",
                            "No suitable method was found."
                        )
                    )

                    return

                utility = recommendation[
                    "predicted_utility"
                ]

                self.method_var.set(
                    method
                )

                self.recommendation_var.set(
                    f"Recommended Method: {method}    "
                    f"Predicted Utility: {utility:.4f}"
                )

            except Exception as error:

                messagebox.showerror(
                    "AUTO Recommendation Error",
                    str(error)
                )

                return


        # ----------------------------------------------------
        # Preprocess cover
        # ----------------------------------------------------

        try:

            processed_cover = (
                self.preprocess_cover()
            )

        except Exception as error:

            messagebox.showerror(
                "Cover Processing Error",
                str(error)
            )

            return


        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        output_dir = Path(
            "output/gui"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        payload_name = (
            self.payload_path.stem
        )

        output_path = (
            output_dir /
            f"{method.lower()}_stego_{payload_name}.png"
        )


        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        try:

            start_time = time.time()


            if method == "LSB":

                lsb.embed_payload(
                    str(processed_cover),
                    str(self.payload_path),
                    str(output_path)
                )


            elif method == "DCT":

                dct.embed_payload(
                    str(processed_cover),
                    str(self.payload_path),
                    str(output_path)
                )


            elif method == "DWT":

                dwt.embed_payload(
                    str(processed_cover),
                    str(self.payload_path),
                    str(output_path)
                )


            else:

                messagebox.showerror(
                    "Invalid Method",
                    f"Unknown method: {method}"
                )

                return


            embedding_time = (
                time.time() - start_time
            )


            # ------------------------------------------------
            # Check output
            # ------------------------------------------------

            if not output_path.exists():

                messagebox.showerror(
                    "Embedding Failed",
                    "The stego image was not created."
                )

                return


            # ------------------------------------------------
            # Read images
            # ------------------------------------------------

            cover_image = cv2.imread(
                str(processed_cover),
                cv2.IMREAD_UNCHANGED
            )

            stego_image = cv2.imread(
                str(output_path),
                cv2.IMREAD_UNCHANGED
            )


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = self.calculate_metrics(
                cover_image,
                stego_image
            )


            # ------------------------------------------------
            # Save state
            # ------------------------------------------------

            self.last_stego_path = (
                output_path
            )

            self.last_method = method

            self.last_metrics = metrics


            # ------------------------------------------------
            # Success popup
            # ------------------------------------------------

            messagebox.showinfo(

                "Data Hidden Successfully",

                f"Method: {method}\n\n"

                f"Stego Image:\n"
                f"{output_path}\n\n"

                f"Embedding Time: "
                f"{embedding_time:.4f} seconds\n\n"

                f"MSE: "
                f"{metrics['mse']:.6f}\n"

                f"PSNR: "
                f"{metrics['psnr']:.4f} dB\n"

                f"SSIM: "
                f"{metrics['ssim']:.4f}\n\n"

                f"Changed Pixels: "
                f"{metrics['changed_percentage']:.4f}%"
            )


        except Exception as error:

            messagebox.showerror(
                "Embedding Error",
                str(error)
            )


    # ========================================================
    # EXTRACT DATA
    # ========================================================

    def extract_data_placeholder(self):

        # ----------------------------------------------------
        # Check stego image
        # ----------------------------------------------------

        if self.last_stego_path is None:

            messagebox.showwarning(
                "No Stego Image",
                "Please hide data first."
            )

            return


        if not self.last_stego_path.exists():

            messagebox.showwarning(
                "Stego Image Not Found",
                "The generated stego image could not be found."
            )

            return


        # ----------------------------------------------------
        # Check method
        # ----------------------------------------------------

        if self.last_method is None:

            messagebox.showwarning(
                "Method Not Found",
                "The embedding method could not be determined."
            )

            return


        method = self.last_method


        # ----------------------------------------------------
        # Extraction directory
        # ----------------------------------------------------

        output_dir = Path(
            "output/gui/extracted"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        try:

            start_time = time.time()


            if method == "LSB":

                result = lsb.extract_payload(
                    str(self.last_stego_path),
                    str(output_dir)
                )


            elif method == "DCT":

                result = dct.extract_payload(
                    str(self.last_stego_path),
                    str(output_dir)
                )


            elif method == "DWT":

                result = dwt.extract_payload(
                    str(self.last_stego_path),
                    str(output_dir)
                )


            else:

                messagebox.showerror(
                    "Invalid Method",
                    f"Unknown method: {method}"
                )

                return


            extraction_time = (
                time.time() - start_time
            )


            # ------------------------------------------------
            # Determine displayed result
            # ------------------------------------------------

            if result is not None:

                extracted_path = result

            else:

                extracted_path = output_dir


            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            messagebox.showinfo(

                "Data Extracted Successfully",

                f"Method: {method}\n\n"

                f"Extracted Payload:\n"
                f"{extracted_path}\n\n"

                f"Extraction Time: "
                f"{extraction_time:.4f} seconds"
            )


        except Exception as error:

            messagebox.showerror(
                "Extraction Error",
                str(error)
            )


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()

    app = SteganographyApp(
        root
    )

    root.mainloop()


if __name__ == "__main__":

    main()