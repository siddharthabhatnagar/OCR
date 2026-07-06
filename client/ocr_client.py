"""
Drop-in Python client for the Docling OCR API (Vercel).

Same API shape as the Unlimited-OCR client, so you can swap backends
without changing your RAG pipeline code.

Usage
-----
    from client.ocr_client import UnlimitedOcrClient, to_langchain_documents

    client = UnlimitedOcrClient(base_url="https://your-app.vercel.app")
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx


# --------------------------------------------------------------------- #
# Response models (dataclasses — no pydantic dependency on client side)
# --------------------------------------------------------------------- #
@dataclass
class BoundingBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Element:
    type: str
    content: str
    page_index: int
    bbox: Optional[BoundingBox] = None
    extra: dict = field(default_factory=dict)


@dataclass
class PageMeta:
    page_index: int
    width_px: int
    height_px: int
    dpi: int


@dataclass
class PageResult:
    page: PageMeta
    markdown: str
    elements: list[Element]


@dataclass
class OcrResponse:
    markdown: str
    pages: list[PageResult]
    total_pages: int
    elapsed_ms: int
    model: str
    warnings: list[str] = field(default_factory=list)


def _parse_response(data: dict) -> OcrResponse:
    pages = []
    for p in data.get("pages", []):
        meta = p["page"]
        elements = []
        for e in p.get("elements", []):
            bbox = None
            if e.get("bbox"):
                bbox = BoundingBox(**e["bbox"])
            elements.append(
                Element(
                    type=e["type"],
                    content=e["content"],
                    page_index=e["page_index"],
                    bbox=bbox,
                    extra=e.get("extra", {}),
                )
            )
        pages.append(
            PageResult(
                page=PageMeta(**meta),
                markdown=p["markdown"],
                elements=elements,
            )
        )
    return OcrResponse(
        markdown=data["markdown"],
        pages=pages,
        total_pages=data["total_pages"],
        elapsed_ms=data["elapsed_ms"],
        model=data["model"],
        warnings=data.get("warnings", []),
    )


# --------------------------------------------------------------------- #
# Sync client
# --------------------------------------------------------------------- #
class OcrClient:
    """Synchronous client. Use as a context manager for clean shutdown."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,  # Docling is fast: 2-5 sec per page typical
    ):
        self.base_url = base_url.rstrip("/")
        headers = {"User-Agent": "docling-ocr-client/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def health(self) -> dict:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def ocr_pdf(self, path: str | os.PathLike) -> OcrResponse:
        return self._ocr_file(path, "application/pdf")

    def ocr_image(self, path: str | os.PathLike) -> OcrResponse:
        path = Path(path)
        ct = _guess_image_mime(path.suffix.lower())
        return self._ocr_file(path, ct)

    def ocr_bytes(
        self,
        data: bytes,
        filename: str = "upload.pdf",
        content_type: str = "application/pdf",
    ) -> OcrResponse:
        r = self._client.post("/ocr", files={"file": (filename, data, content_type)})
        r.raise_for_status()
        return _parse_response(r.json())

    def _ocr_file(self, path: str | os.PathLike, content_type: str) -> OcrResponse:
        path = Path(path)
        with path.open("rb") as f:
            r = self._client.post(
                "/ocr",
                files={"file": (path.name, f, content_type)},
            )
        r.raise_for_status()
        return _parse_response(r.json())

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Backward-compat alias for code written against the Unlimited-OCR client
UnlimitedOcrClient = OcrClient


# --------------------------------------------------------------------- #
# Async client
# --------------------------------------------------------------------- #
class AsyncOcrClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        headers = {"User-Agent": "docling-ocr-client/0.1"}
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
        return _parse_response(r.json())

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
# CLI smoke test
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Smoke test the OCR service.")
    p.add_argument("url", help="Service base URL")
    p.add_argument("file", help="PDF or image to OCR")
    p.add_argument("--api-key", default=os.getenv("OCR_API_KEY"))
    args = p.parse_args()

    with OcrClient(args.url, api_key=args.api_key) as client:
        h = client.health()
        print("Health:", json.dumps(h, indent=2))
        ext = Path(args.file).suffix.lower()
        result = client.ocr_pdf(args.file) if ext == ".pdf" else client.ocr_image(args.file)
        print(f"\nOCR done in {result.elapsed_ms} ms | {result.total_pages} page(s) | model={result.model}")
        print("\n=== Markdown (first 800 chars) ===\n")
        print(result.markdown[:800])
        print("\n=== Per-page element summary ===")
        for page in result.pages:
            print(f"\n-- Page {page.page.page_index} --")
            for el in page.elements:
                preview = el.content.replace("\n", " ")[:80]
                print(f"  [{el.type:9}] {preview}")
