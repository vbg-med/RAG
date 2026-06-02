import base64
import requests
import logging
import io
from pathlib import Path
from pypdf import PdfReader
from PIL import Image

try:
    from llama_index.core import Document
except ImportError:
    from llama_index import Document
from src.config import Config

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extracts images from PDFs and captions them using local multimodal LLM (LLaVA)"""

    def __init__(self, config: Config):
        self.config = config

    def process_pdf(self, pdf_path: str):
        """Extract images from PDF and return list of Document objects with captions"""
        documents = []
        if not self.config.USE_IMAGE_CAPTIONS:
            logger.info("Image captioning is disabled in configuration.")
            return documents

        try:
            logger.info(f"Extracting images from {pdf_path}...")
            reader = PdfReader(pdf_path)
            pdf_name = Path(pdf_path).name

            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                images = page.images

                for img_idx, image_file_object in enumerate(images):
                    img_name = f"{Path(pdf_path).stem}_p{page_num}_img{img_idx}.png"
                    save_path = self.config.DATA_DIR / "processed" / img_name

                    try:
                        # Open bytes using PIL and save to disk
                        image = Image.open(io.BytesIO(image_file_object.data))
                        image.save(save_path, format="PNG")
                        logger.info(f"Extracted and saved image: {save_path.name}")

                        # Generate description / caption
                        caption = self._caption_image(str(save_path))
                        if caption:
                            doc = Document(
                                text=f"Visual content on page {page_num} of {pdf_name}: {caption}",
                                metadata={
                                    "source": pdf_name,
                                    "page": str(page_num),
                                    "type": "image",
                                    "image_path": str(save_path),
                                },
                            )
                            documents.append(doc)
                    except Exception as img_err:
                        logger.error(
                            f"Error processing image {img_idx} on page {page_num}: {img_err}"
                        )

        except Exception as e:
            logger.error(f"Error processing PDF images: {e}")

        return documents

    def _caption_image(self, image_path: str) -> str:
        """Caption an image using Ollama LLaVA"""
        try:
            logger.info(
                f"Captioning image {image_path} with {self.config.IMAGE_MODEL}..."
            )

            # Read image and convert to base64
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # 1. Attempt using Python client library
            try:
                import ollama

                response = ollama.generate(
                    model=self.config.IMAGE_MODEL,
                    prompt="Describe this image in detail. Focus on charts, tables, diagrams, or text.",
                    images=[image_bytes],
                )
                caption = response.get("response", "").strip()
                if caption:
                    return caption
            except Exception as lib_err:
                logger.warning(
                    f"Ollama Python library call failed: {lib_err}. Trying direct HTTP request..."
                )

            # 2. Fallback to direct HTTP request (more robust under various package versions)
            url = f"{self.config.OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": self.config.IMAGE_MODEL,
                "prompt": "Describe this image in detail. Focus on charts, tables, diagrams, or text.",
                "images": [image_b64],
                "stream": False,
            }
            res = requests.post(url, json=payload, timeout=60)
            if res.status_code == 200:
                caption = res.json().get("response", "").strip()
                return caption
            else:
                logger.error(
                    f"Ollama HTTP API returned error status {res.status_code}: {res.text}"
                )

        except Exception as e:
            logger.error(f"Error captioning image {image_path}: {e}")

        return ""
