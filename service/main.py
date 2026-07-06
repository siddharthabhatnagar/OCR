"""
FastAPI app with /ocr, /ocr/raw, /health endpoints.

On HF Spaces ZeroGPU, the model is NOT loaded at startup (no GPU at startup).
It loads lazily on first OCR call inside a @spaces.GPU function.

This module is imported by app.py (the Gradio entry point). app.py mounts
the FastAPI routes onto the Gradio Blocks' underlying FastAPI app.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from .config import Settings, get_settings
from .model import get_model_manager
from .ocr import ocr_image, ocr_pdf
from .schemas import HealthResponse, OcrResponse


_startup_ts = time.time()


app = FastAPI(
    title="Unlimited-OCR API",
    description=(
        "FastAPI wrapper around baidu/Unlimited-OCR for RAG ingestion. "
        "Accepts PDF or image uploads, returns per-page markdown plus "
        "typed elements (text / heading / table / equation / figure)."
    ),
    version="0.1.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serialize GPU access — single GPU, single worker, no concurrent forward passes.
_gate: asyncio.Semaphore | None = None


def _get_gate() -> asyncio.Semaphore:
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(get_settings().max_concurrent_requests)
    return _gate


# --------------------------------------------------------------------- #
# Auth (optional). If API_KEY env var is set, clients must send Bearer token.
# --------------------------------------------------------------------- #
_security = HTTPBearer(
    scheme_name="APIKey",
    description="Paste your API_KEY value here (without 'Bearer ' prefix).",
    auto_error=False,
)


def _verify_auth(
    settings: Settings = Depends(get_settings),
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> None:
    if not settings.api_key:
        return
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Click Authorize at top-right.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization scheme must be Bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if creds.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
async def root():
    s = get_settings()
    return {
        "service": "unlimited-ocr-api",
        "version": app.version,
        "model": s.model_id,
        "docs": "/docs",
        "endpoints": ["/health", "/ocr", "/ocr/raw"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    s = get_settings()
    mm = get_model_manager()
    # On ZeroGPU, "not loaded" is the expected steady state — the model
    # loads on first OCR call inside @spaces.GPU.
    if mm.load_error:
        status_str = "error"
    elif mm.is_loaded:
        status_str = "ok"
    elif mm.needs_zero_gpu:
        status_str = "ok"
    else:
        status_str = "loading"
    return HealthResponse(
        status=status_str,  # type: ignore[arg-type]
        model_loaded=mm.is_loaded,
        device=mm.device,
        hf_space=s.is_hf_space,
        uptime_s=int(time.time() - _startup_ts),
    )


@app.post(
    "/ocr",
    response_model=OcrResponse,
    tags=["ocr"],
    dependencies=[Depends(_verify_auth)],
)
async def ocr_upload(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """OCR an uploaded PDF or image. Click Authorize at top-right first."""
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")

    name = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    is_pdf = name.endswith(".pdf") or "pdf" in content_type
    is_image = (
        any(ext in name for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        or content_type.startswith("image/")
    )
    if not (is_pdf or is_image):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type or name}",
        )

    async with _get_gate():
        loop = asyncio.get_event_loop()
        try:
            if is_pdf:
                result = await loop.run_in_executor(None, ocr_pdf, payload, settings)
            else:
                result = await loop.run_in_executor(None, ocr_image, payload, settings)
        except Exception as e:
            logger.exception("OCR failed")
            raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post(
    "/ocr/raw",
    response_model=OcrResponse,
    tags=["ocr"],
    dependencies=[Depends(_verify_auth)],
)
async def ocr_raw_image(
    settings: Settings = Depends(get_settings),
    content_type: str = Header(default="image/png"),
    body: bytes = b"",
):
    """OCR a raw image sent as the request body (no multipart)."""
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Expected image/* content type, got {content_type}",
        )
    async with _get_gate():
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, ocr_image, body, settings)
        except Exception as e:
            logger.exception("OCR failed")
            raise HTTPException(status_code=500, detail=str(e))
    return result
