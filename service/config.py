"""
Configuration. All values overridable via env vars (HF Spaces injects these).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model
    model_id: str = "baidu/Unlimited-OCR"
    model_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    trust_remote_code: bool = True

    # Generation (passed to model.infer_multi)
    max_new_tokens: int = 16384

    # PDF pipeline
    pdf_dpi: int = 144
    pdf_max_pages: int = 50

    # Chunk size for infer_multi — model can stitch many pages in one call,
    # but we chunk to stay under ZeroGPU's 5-min per-call limit.
    chunk_size: int = 4  # pages per infer_multi call

    # Server
    port: int = int(os.getenv("PORT", "7860"))
    max_concurrent_requests: int = 4

    # Auth (optional). If set, clients must send `Authorization: Bearer <key>`.
    api_key: str | None = None

    @property
    def is_hf_space(self) -> bool:
        return os.getenv("SPACE_AUTHOR_NAME") is not None

    @property
    def hf_token(self) -> str | None:
        return os.getenv("HF_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()
