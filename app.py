from pathlib import Path
import tempfile

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.config import load_config
from src.infer import build_for_inference, predict

app = FastAPI(
    title="Multimodal VQA API",
    version="1.0.0",
    description="Image + question + four options Visual Question Answering API."
)

CFG = load_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "fusion"

MODEL = None
MODEL_ERROR = None


def get_model():
    global MODEL, MODEL_ERROR
    if MODEL is not None:
        return MODEL
    try:
        MODEL = build_for_inference(MODEL_NAME, CFG, DEVICE)
        return MODEL
    except Exception as exc:
        MODEL_ERROR = str(exc)
        raise


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_ERROR is None else "model_not_loaded",
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "error": MODEL_ERROR
    }


@app.post("/v1/qa")
async def visual_question_answering(
    image: UploadFile = File(...),
    question: str = Form(...),
    options: str = Form(...)
):
    if image.content_type not in {
        "image/jpeg", "image/png", "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Image must be JPEG, PNG or WEBP."
        )

    import json
    try:
        parsed_options = json.loads(options)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="options must be a JSON array of four strings."
        )

    if (
        not isinstance(parsed_options, list)
        or len(parsed_options) != 4
        or not all(isinstance(x, str) and x.strip() for x in parsed_options)
    ):
        raise HTTPException(
            status_code=400,
            detail="options must contain exactly four non-empty strings."
        )

    data = await image.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image is larger than 10 MB.")

    try:
        from PIL import Image
        import io
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
        model = get_model()
        result = predict(
            model,
            MODEL_NAME,
            pil_image,
            question,
            parsed_options,
            DEVICE
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}"
        )
