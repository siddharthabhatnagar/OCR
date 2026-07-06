"""
Pydantic response schemas — matches the Unlimited-OCR API shape exactly
so the same RAG client works with both.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ElementType = Literal["text", "table", "equation", "figure", "heading", "list"]


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class PageMeta(BaseModel):
    page_index: int
    width_px: int
    height_px: int
    dpi: int


class Element(BaseModel):
    type: ElementType
    content: str
    page_index: int
    bbox: Optional[BoundingBox] = None
    extra: dict = Field(default_factory=dict)


class PageResult(BaseModel):
    page: PageMeta
    markdown: str
    elements: list[Element]


class OcrResponse(BaseModel):
    markdown: str
    pages: list[PageResult]
    total_pages: int
    elapsed_ms: int
    model: str
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    device: str
    uptime_s: int
