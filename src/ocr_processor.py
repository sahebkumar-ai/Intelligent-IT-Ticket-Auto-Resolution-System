
import os
from pathlib import Path
from typing import Union
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

class OCRProcessor:

    def __init__(
        self,
        tesseract_cmd: str | None = None
    ):
        configured_cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")
        if not configured_cmd:
            windows_install = Path(
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            if windows_install.is_file():
                configured_cmd = str(windows_install)

        if configured_cmd:
            pytesseract.pytesseract.tesseract_cmd = configured_cmd

    def load_image(
        self,
        image_source: Union[str, Path, Image.Image]
    ) -> Image.Image:
        

        if isinstance(
            image_source,
            Image.Image
        ):
            return image_source

        image_path = Path(
            image_source
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        try:
            return Image.open(
                image_path
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to open image: {exc}"
            ) from exc

    def preprocess_image(
        self,
        image: Image.Image
    ) -> Image.Image:
        

        image = image.convert(
            "L"
        )

        contrast = ImageEnhance.Contrast(
            image
        )

        image = contrast.enhance(
            2.0
        )

        image = image.filter(
            ImageFilter.SHARPEN
        )

        return image

    def extract_text(
        self,
        image_source: Union[str, Path, Image.Image]
    ) -> str:
        

        image = self.load_image(
            image_source
        )

        processed_image = (
            self.preprocess_image(
                image
            )
        )

        try:
            text = pytesseract.image_to_string(
                processed_image
            )
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract OCR executable was not found. Install Tesseract OCR, "
                "then set TESSERACT_CMD to the full path of tesseract.exe "
                "(for example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe)."
            )

        return self.clean_ocr_text(
            text
        )

    @staticmethod
    def clean_ocr_text(
        text: str
    ) -> str:
        

        if not text:
            return ""

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        return "\n".join(
            lines
        )

    def extract_from_uploaded_file(
        self,
        uploaded_file
    ) -> str:
        

        if uploaded_file is None:
            return ""

        try:
            image = Image.open(
                uploaded_file
            )

            return self.extract_text(
                image
            )

        except Exception as exc:
            raise ValueError(
                f"Unable to process uploaded image: "
                f"{exc}"
            ) from exc


def extract_text_from_image(
    image_source
) -> str:
    
    processor = OCRProcessor()

    return processor.extract_text(
        image_source
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python src/ocr_processor.py "
            "path/to/screenshot.png"
        )

        sys.exit(0)

    image_path = sys.argv[1]

    processor = OCRProcessor()

    try:

        extracted_text = (
            processor.extract_text(
                image_path
            )
        )

        print(
            "\nExtracted Text"
        )

        print(
            "=" * 50
        )

        print(
            extracted_text
        )

    except Exception as exc:

        print(
            f"OCR Error: {exc}"
        )

