"""
Integration smoke tests. Run against a live Vercel deployment:

    OCR_BASE_URL=https://your-app.vercel.app pytest tests/test_api.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

BASE = os.getenv("OCR_BASE_URL", "http://localhost:3000").rstrip("/")
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", "samples"))


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE, timeout=120) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    r.raise_for_status()
    data = r.json()
    assert data["status"] in {"ok", "loading"}
    assert data["device"] == "cpu"


def test_root(client):
    r = client.get("/")
    r.raise_for_status()
    data = r.json()
    assert data["service"] == "docling-ocr-api"


@pytest.mark.skipif(
    not SAMPLES_DIR.exists(),
    reason="Put a sample PDF/PNG in samples/ to run this test",
)
def test_ocr_pdf(client):
    pdfs = sorted(SAMPLES_DIR.glob("*.pdf"))
    assert pdfs, "Put a sample PDF in samples/"
    pdf = pdfs[0]
    with pdf.open("rb") as f:
        r = client.post(
            "/ocr",
            files={"file": (pdf.name, f, "application/pdf")},
        )
    r.raise_for_status()
    data = r.json()
    assert data["total_pages"] >= 1
    assert data["markdown"]
    assert data["model"] == "docling"
