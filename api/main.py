
from pathlib import Path
import sys
import time
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR)
)
from src.ticket_pipeline import TicketPipeline
from src.ocr_processor import OCRProcessor

app = FastAPI(
    title="Intelligent IT Ticket Auto-Resolution API",
    description=(
        "Production-oriented AI API for enterprise "
        "IT ticket classification and solution recommendation."
    ),
    version="1.0.0",
)
pipeline = None
ocr_processor = None


def get_pipeline():
    
    global pipeline

    if pipeline is None:

        pipeline = TicketPipeline(
            confidence_threshold=0.80
        )

    return pipeline


def get_ocr_processor():
    

    global ocr_processor

    if ocr_processor is None:

        ocr_processor = OCRProcessor()

    return ocr_processor

class TicketRequest(BaseModel):
    

    ticket_text: str = Field(
        ...,
        min_length=3,
        description="IT support ticket description"
    )

    ocr_text: Optional[str] = Field(
        default="",
        description="Text extracted from a screenshot"
    )

    log_text: Optional[str] = Field(
        default="",
        description="System or application logs"
    )

    ticket_id: Optional[str] = Field(
        default=None,
        description="Optional ticket identifier"
    )

    confidence_threshold: Optional[float] = Field(
        default=0.80,
        ge=0.50,
        le=0.99,
        description="Minimum confidence for automation"
    )
class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str
    service: str
    version: str

@app.get("/")
def root():
    return {
        "service": (
            "Intelligent IT Ticket "
            "Auto-Resolution API"
        ),
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs"
    }

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():
    return {
        "status": "healthy",
        "service": "it-ticket-auto-resolution",
        "version": "1.0.0"
    }

@app.post("/ticket/analyze")
def analyze_ticket(
    request: TicketRequest
):

    start_time = time.perf_counter()

    try:

        ai_pipeline = get_pipeline()
        ai_pipeline.confidence_threshold = (
            request.confidence_threshold
        )

        result = ai_pipeline.process_ticket(
            ticket_text=request.ticket_text,
            ocr_text=request.ocr_text or "",
            log_text=request.log_text or "",
            ticket_id=request.ticket_id
        )

        processing_time = (
            time.perf_counter()
            - start_time
        )

        result["api"] = {
            "processing_time_seconds": round(
                processing_time,
                4
            )
        }

        return result

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "AI model is not available. "
                "Train the model first."
            )
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc
@app.post("/predict")
def predict_ticket(
    request: TicketRequest
):
    try:

        ai_pipeline = get_pipeline()

        processed_text = (
            ai_pipeline.classifier
            .vectorizer
        )
        from src.preprocessing import (
            combine_ticket_text
        )

        cleaned_text = combine_ticket_text(
            ticket_text=request.ticket_text,
            ocr_text=request.ocr_text or "",
            log_text=request.log_text or ""
        )

        prediction = (
            ai_pipeline.classifier.predict(
                cleaned_text
            )
        )

        return {
            "success": True,
            "category": prediction[
                "category"
            ],
            "confidence": prediction[
                "confidence"
            ]
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc
@app.post("/ocr")
async def process_screenshot(
    file: UploadFile = File(...)
):
    
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type could not be determined."
        )

    if not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        from io import BytesIO

        from PIL import Image

        image = Image.open(
            BytesIO(image_bytes)
        )

        processor = get_ocr_processor()

        extracted_text = (
            processor.extract_text(
                image
            )
        )

        return {
            "success": True,
            "filename": file.filename,
            "extracted_text": extracted_text
        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"OCR processing failed: {exc}"
            )
        ) from exc
@app.post("/ticket/analyze-image")
async def analyze_ticket_with_image(
    ticket_text: str = "",
    ticket_id: Optional[str] = None,
    file: UploadFile = File(...)
):
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type could not be determined."
        )

    if not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        from io import BytesIO

        from PIL import Image

        image = Image.open(
            BytesIO(image_bytes)
        )

        processor = get_ocr_processor()

        extracted_text = (
            processor.extract_text(
                image
            )
        )

        ai_pipeline = get_pipeline()

        result = ai_pipeline.process_ticket(
            ticket_text=ticket_text,
            ocr_text=extracted_text,
            ticket_id=ticket_id
        )

        result["ocr"] = {
            "filename": file.filename,
            "extracted_text": extracted_text
        }

        return result

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc
@app.get("/system/info")
def system_info():
    

    return {
        "system": (
            "Intelligent IT Ticket "
            "Auto-Resolution System"
        ),
        "classification": {
            "method": "TF-IDF + Logistic Regression",
            "target_accuracy": ">= 80%"
        },
        "solution_retrieval": {
            "method": "Knowledge Base Lookup"
        },
        "ocr": {
            "enabled": True
        },
        "automation": {
            "default_confidence_threshold": 0.80
        },
        "performance": {
            "target_response_time": "< 2 seconds"
        }
    }
