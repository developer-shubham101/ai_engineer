"""Vision processing providers."""

import logging

from .interfaces import IVisionProvider, ProcessingResult

logger = logging.getLogger(__name__)


class TesseractOCRProvider(IVisionProvider):
    """Tesseract OCR provider."""

    async def extract_text(self, image_path: str) -> ProcessingResult:
        """Extract text from image using Tesseract."""
        try:
            import pytesseract
            from PIL import Image

            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            # Check if Tesseract is installed
            try:
                pytesseract.get_tesseract_version()
            except pytesseract.TesseractNotFoundError:
                return ProcessingResult(
                    success=False,
                    data={},
                    error="Tesseract OCR not installed. Install: Windows: 'choco install tesseract' or download from https://github.com/UB-Mannheim/tesseract/wiki. Linux: 'sudo apt install tesseract-ocr'. Then: pip install pytesseract pillow"
                )

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)

            return ProcessingResult(
                success=True,
                data={
                    "text": text.strip(),
                    "provider": "tesseract",
                    "confidence": 0.8
                }
            )

        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="pytesseract not installed. Run: pip install pytesseract pillow"
            )
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )

    async def describe_image(self, image_path: str) -> ProcessingResult:
        """Basic image description (placeholder)."""
        try:
            from PIL import Image

            image = Image.open(image_path)
            width, height = image.size
            mode = image.mode

            description = f"Image: {width}x{height} pixels, {mode} mode"

            return ProcessingResult(
                success=True,
                data={
                    "description": description,
                    "provider": "basic",
                    "width": width,
                    "height": height,
                    "mode": mode
                }
            )

        except Exception as e:
            logger.error(f"Image description error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )

    def get_provider_name(self) -> str:
        return "tesseract"


class PaddleOCRProvider(IVisionProvider):
    """PaddleOCR provider (better accuracy)."""

    _instance = None
    _ocr = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialization handled in extract_text method
        pass

    async def extract_text(self, image_path: str) -> ProcessingResult:
        """Extract text using PaddleOCR."""
        try:
            from paddleocr import PaddleOCR

            if not PaddleOCRProvider._initialized:
                PaddleOCRProvider._ocr = PaddleOCR(use_angle_cls=True, lang='en')
                PaddleOCRProvider._initialized = True

            result = PaddleOCRProvider._ocr.ocr(image_path, cls=True)

            text_lines = []
            for line in result:
                for word_info in line:
                    text_lines.append(word_info[1][0])

            text = '\n'.join(text_lines)

            return ProcessingResult(
                success=True,
                data={
                    "text": text.strip(),
                    "provider": "paddleocr",
                    "confidence": 0.9
                }
            )

        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="PaddleOCR not installed. Run: pip install paddlepaddle paddleocr"
            )
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )

    async def describe_image(self, image_path: str) -> ProcessingResult:
        """Basic image description."""
        return await TesseractOCRProvider().describe_image(image_path)

    def get_provider_name(self) -> str:
        return "paddleocr"


class CLIPProvider(IVisionProvider):
    """CLIP provider for image-text similarity."""

    _model = None
    _processor = None
    _initialized = False

    async def extract_text(self, image_path: str) -> ProcessingResult:
        """CLIP doesn't extract text directly."""
        # return await self.describe_image(image_path=image_path)
        return ProcessingResult(
            success=False,
            data={},
            error="CLIP is for image-text similarity, not text extraction"
        )

    async def describe_image(self, image_path: str, text_queries: list = None) -> ProcessingResult:
        """Get image-text similarity scores."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
            import torch

            if not CLIPProvider._initialized:
                CLIPProvider._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                CLIPProvider._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                CLIPProvider._initialized = True

            image = Image.open(image_path)
            
            if not text_queries:
                text_queries = ["a photo", "a person", "an object", "text document", "nature scene"]

            inputs = CLIPProvider._processor(text=text_queries, images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = CLIPProvider._model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            results = {query: float(prob) for query, prob in zip(text_queries, probs[0])}
            best_match = max(results, key=results.get)

            return ProcessingResult(
                success=True,
                data={
                    "similarities": results,
                    "best_match": best_match,
                    "confidence": results[best_match],
                    "provider": "clip"
                }
            )

        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="CLIP dependencies not installed. Run: pip install transformers torch pillow"
            )
        except Exception as e:
            logger.error(f"CLIP error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )

    def get_provider_name(self) -> str:
        return "clip"


class YOLOProvider(IVisionProvider):
    """YOLO provider for object detection."""

    _model = None
    _initialized = False

    async def extract_text(self, image_path: str) -> ProcessingResult:
        """YOLO doesn't extract text."""
        return ProcessingResult(
            success=False,
            data={},
            error="YOLO is for object detection, not text extraction"
        )

    async def describe_image(self, image_path: str) -> ProcessingResult:
        """Detect objects in image."""
        try:
            import ultralytics
            from ultralytics import YOLO

            if not YOLOProvider._initialized:
                YOLOProvider._model = YOLO('yolov8n.pt')  # Nano model for CPU
                YOLOProvider._initialized = True

            results = YOLOProvider._model(image_path, verbose=False)
            
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = r.names[cls_id]
                    
                    detections.append({
                        "object": name,
                        "confidence": conf,
                        "bbox": box.xyxy[0].tolist()
                    })

            description = f"Detected {len(detections)} objects: " + ", ".join([d["object"] for d in detections[:5]])

            return ProcessingResult(
                success=True,
                data={
                    "detections": detections,
                    "description": description,
                    "provider": "yolo",
                    "count": len(detections)
                }
            )

        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="YOLO not installed. Run: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"YOLO error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )

    def get_provider_name(self) -> str:
        return "yolo"


class BLIPProvider(IVisionProvider):
    """BLIP provider for image captioning."""

    _model = None
    _processor = None
    _initialized = False

    async def extract_text(self, image_path: str) -> ProcessingResult:
        """BLIP doesn't extract text directly."""
        return ProcessingResult(
            success=False,
            data={},
            error="BLIP is for image captioning, not text extraction"
        )

    async def describe_image(self, image_path: str) -> ProcessingResult:
        """Generate image caption."""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image
            import torch

            if not BLIPProvider._initialized:
                BLIPProvider._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                BLIPProvider._model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                BLIPProvider._initialized = True

            image = Image.open(image_path).convert('RGB')
            inputs = BLIPProvider._processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = BLIPProvider._model.generate(**inputs, max_length=50)
                caption = BLIPProvider._processor.decode(out[0], skip_special_tokens=True)

            return ProcessingResult(
                success=True,
                data={
                    "description": caption,
                    "caption": caption,
                    "provider": "blip",
                    "confidence": 0.85
                }
            )

        except ImportError:
            return ProcessingResult(
                success=False,
                data={},
                error="BLIP dependencies not installed. Run: pip install transformers torch pillow"
            )
        except Exception as e:
            logger.error(f"BLIP error: {e}")
            return ProcessingResult(
                success=False,
                data={},
                error=str(e)
            )

    def get_provider_name(self) -> str:
        return "blip"


# Factory function
def create_vision_provider(provider_name: str = "tesseract") -> IVisionProvider:
    providers = {
        "tesseract": TesseractOCRProvider,
        "paddleocr": PaddleOCRProvider,
        "clip": CLIPProvider,
        "yolo": YOLOProvider,
        "blip": BLIPProvider
    }

    provider_class = providers.get(provider_name, TesseractOCRProvider)
    return provider_class()
