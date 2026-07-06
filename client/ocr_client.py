"""
Drop-in Python client for the Unlimited-OCR service.

Usage
-----
    from client.ocr_client import UnlimitedOcrClient

    client = UnlimitedOcrClient(base_url="https://your-space.hf.space")
    result = client.ocr_pdf("paper.pdf")

    print(result.markdown)               # full document markdown
    for page in result.pages:            # per-page typed elements
        for el in page.elements:
            print(el.type, el.content[:80])

RAG integration helpers:
    to_langchain_documents(result)       # one Document per page
    to_llamaindex_nodes(result)          # one TextNode per Element
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import httpx


# --------------------------------------------------------------------- #
# Response models (mirror server-side schemas)
# --------------------------------------------------------------------- #
class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


class Element:
    def __init__(self, type, content, page_index, bbox=None, confidence=None, extra=None):
        self.type = type
        self.content = content
        self.page_index = page_index
        self.bbox = BoundingBox(**bbox) if bbox else None
        self.confidence = confidence
        self.extra = extra or {}


class PageMeta:
    def __init__(self, page_index, width_px, height_px, dpi):
        self.page_index = page_index
        self.width_px = width_px
        self.height_px = height_px
        self.dpi = dpi


class PageResult:
    def __init__(self, page, markdown, elements):
        self.page = PageMeta(**page)
        self.markdown = markdown
        self.elements = [Element(**e) for e in elements]


class OcrResponse:
    def __init__(self, markdown, pages, total_pages, elapsed_ms, model, warnings=None):
        self.markdown = markdown
        self.pages = [PageResult(**p) for p in pages]
        self.total_pages = total_pages
        self.elapsed_ms = elapsed_ms
        self.model = model
        self.warnings = warnings or []


# --------------------------------------------------------------------- #
# Sync client
# --------------------------------------------------------------------- #
class UnlimitedOcrClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 600.0,  # 10 min for cold start
    ):
        self.base_url = base_url.rstrip("/")
        headers = {"User-Agent": "unlimited-ocr-client/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def health(self) -> dict:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def ocr_pdf(self, path: str | os.PathLike) -> OcrResponse:
        path = Path(path)
        with path.open("rb") as f:
            r = self._client.post(
                "/ocr",
                files={"file": (path.name, f, "application/pdf")},
            )
        r.raise_for_status()
        return OcrResponse(**r.json())

    def ocr_image(self, path: str | os.PathLike) -> OcrResponse:
        path = Path(path)
        ct = _guess_image_mime(path.suffix.lower())
        with path.open("rb") as f:
            r = self._client.post("/ocr", files={"file": (path.name, f, ct)})
        r.raise_for_status()
        return OcrResponse(**r.json())

    def ocr_bytes(
        self,
        data: bytes,
        filename: str = "upload.pdf",
        content_type: str = "application/pdf",
    ) -> OcrResponse:
        r = self._client.post("/ocr", files={"file": (filename, data, content_type)})
        r.raise_for_status()
        return OcrResponse(**r.json())

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# --------------------------------------------------------------------- #
# Async client
# --------------------------------------------------------------------- #
class AsyncUnlimitedOcrClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        headers = {"User-Agent": "unlimited-ocr-client/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=timeout
        )

    async def health(self):
        r = await self._client.get("/health")
        r.raise_for_status()
        return r.json()

    async def ocr_pdf(self, path):
        path = Path(path)
        with path.open("rb") as f:
            r = await self._client.post(
                "/ocr",
                files={"file": (path.name, f, "application/pdf")},
            )
        r.raise_for_status()
        return OcrResponse(**r.json())

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()


# --------------------------------------------------------------------- #
# RAG integration helpers
# --------------------------------------------------------------------- #
def to_langchain_documents(result: OcrResponse, source: str = ""):
    """Convert to LangChain Documents, one per page."""
    from langchain_core.documents import Document

    docs = []
    for page in result.pages:
        metadata = {
            "source": source,
            "page": page.page.page_index,
            "width_px": page.page.width_px,
            "height_px": page.page.height_px,
            "model": result.model,
        }
        docs.append(Document(page_content=page.markdown, metadata=metadata))
    return docs


def to_llamaindex_nodes(result: OcrResponse, source: str = ""):
    """Convert to LlamaIndex TextNodes, one per Element."""
    from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

    nodes = []
    parent_id = f"doc:{source or 'unknown'}"
    for page in result.pages:
        page_id = f"{parent_id}:p{page.page.page_index}"
        for i, el in enumerate(page.elements):
            node = TextNode(
                text=el.content,
                id_=f"{page_id}:e{i}",
                metadata={
                    "source": source,
                    "page": page.page.page_index,
                    "element_type": el.type,
                    "model": result.model,
                },
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=parent_id),
                },
            )
            nodes.append(node)
    return nodes


# --------------------------------------------------------------------- #
def _guess_image_mime(suffix: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")


# --------------------------------------------------------------------- #
# CLI smoke test: python client/ocr_client.py URL FILE [--api-key KEY]
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Smoke test the OCR service.")
    p.add_argument("url", help="Service base URL")
    p.add_argument("file", help="PDF or image to OCR")
    p.add_argument("--api-key", default=os.getenv("OCR_API_KEY"))
    args = p.parse_args()

    with UnlimitedOcrClient(args.url, api_key=args.api_key) as client:
        h = client.health()
        print("Health:", json.dumps(h, indent=2))
        ext = Path(args.file).suffix.lower()
        result = client.ocr_pdf(args.file) if ext == ".pdf" else client.ocr_image(args.file)
        print(f"\nOCR done in {result.elapsed_ms} ms | {result.total_pages} page(s)")
        print("\n=== Markdown (first 800 chars) ===\n")
        print(result.markdown[:800])
