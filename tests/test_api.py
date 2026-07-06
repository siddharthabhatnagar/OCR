"""
Integration smoke tests. Run against a live instance:

    OCR_BASE_URL=http://localhost:7860 pytest tests/test_api.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

BASE = os.getenv("OCR_BASE_URL", "http://localhost:7860").rstrip("/")
API_KEY = os.getenv("OCR_API_KEY", "")
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", "samples"))


def _headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE, headers=_headers(), timeout=600) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    r.raise_for_status()
    data = r.json()
    assert data["status"] in {"ok", "loading"}
    assert "device" in data


@pytest.mark.skipif(
    not SAMPLES_DIR.exists(),
    reason="Put a sample PDF/PNG in samples/ to run this test",
)
def test_ocr_pdf(client):
    pdfs = sorted(SAMPLES_DIR.glob("*.pdf"))
    assert pdfs, "Put a sample PDF in samples/"
    pdf = pdfs[0]
    with pdf.open("rb") as f:
        r = client.post("/ocr", files={"file": (pdf.name, f, "application/pdf")})
    r.raise_for_status()
    data = r.json()
    assert data["total_pages"] >= 1
    assert data["markdown"]
    assert any(p["elements"] for p in data["pages"])


@pytest.mark.skipif(
    not SAMPLES_DIR.exists(),
    reason="Put a sample PNG/JPG in samples/ to run this test",
)
def test_ocr_image(client):
    imgs = sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg"))
    assert imgs, "Put a sample image in samples/"
    img = imgs[0]
    ct = "image/png" if img.suffix == ".png" else "image/jpeg"
    with img.open("rb") as f:
        r = client.post("/ocr", files={"file": (img.name, f, ct)})
    r.raise_for_status()
    data = r.json()
    assert data["total_pages"] == 1
