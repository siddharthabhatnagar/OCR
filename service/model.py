"""
Model loader for baidu/Unlimited-OCR on HF Spaces ZeroGPU.

Key design
----------
1. **No GPU work at import time.** The container starts with no GPU; ZeroGPU
   only attaches one inside `@spaces.GPU` decorated functions.

2. **Model loads lazily on first OCR call**, inside a `@spaces.GPU` function.
   The model is downloaded to CPU first, then moved to CUDA.

3. **Uses the model's built-in `infer_multi()` method** (from the official
   README). The model natively handles multi-page stitching via a single
   `<image>` token, and emits `<PAGE>` markers between pages in its output.

4. **`torch.cuda.init()` is called inside `@spaces.GPU`** to refresh PyTorch's
   cached CUDA detection (cached at import time when no GPU existed).

5. **`duration=600`** on `@spaces.GPU` gives us 10 min per call — enough for
   cold-start (download 6 GB + load + infer).

Reference: https://huggingface.co/baidu/Unlimited-OCR
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any

import torch
from loguru import logger

from .config import Settings, get_settings

# `spaces` is only available on HF Spaces. Try-import so the module remains
# importable on a local CPU box for tests.
try:
    import spaces  # type: ignore
    _HAS_SPACES = True
except Exception:
    spaces = None  # type: ignore
    _HAS_SPACES = False


# Official prompt for multi-page parsing. The `<image>` token is REQUIRED.
DEFAULT_PROMPT = "<image>Multi page parsing."


class ModelManager:
    """Thread-safe singleton holding the Unlimited-OCR model + tokenizer."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str:
        if self._loaded and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def needs_zero_gpu(self) -> bool:
        """True when @spaces.GPU should wrap inference."""
        if os.getenv("SPACES_ZERO_GPU", "0") == "1":
            return True
        if _HAS_SPACES and os.getenv("SPACE_AUTHOR_NAME") is not None:
            return True
        return False

    def _resolve_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.settings.model_dtype]

    def load(self) -> None:
        """Idempotent model load. Must be called inside @spaces.GPU on ZeroGPU."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                from transformers import AutoModel, AutoTokenizer

                logger.info(
                    "Loading Unlimited-OCR | model={} dtype={} zero_gpu={}",
                    self.settings.model_id,
                    self.settings.model_dtype,
                    self.needs_zero_gpu,
                )

                dtype = self._resolve_dtype()

                # Tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_id,
                    trust_remote_code=self.settings.trust_remote_code,
                    token=self.settings.hf_token,
                )

                # Model — load to CPU first (works in every env, ZeroGPU or not).
                # The model's config.json registers under AutoModel (NOT
                # AutoModelForCausalLM), so we MUST use AutoModel.
                self._model = AutoModel.from_pretrained(
                    self.settings.model_id,
                    trust_remote_code=self.settings.trust_remote_code,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    token=self.settings.hf_token,
                    device_map="cpu",
                )
                self._model = self._model.eval()

                # If CUDA is available, move there now.
                if torch.cuda.is_available():
                    logger.info("Moving model to CUDA")
                    self._model = self._model.to("cuda")

                self._loaded = True
                self._load_error = None
                logger.info("Model loaded successfully")
            except Exception as e:
                self._load_error = repr(e)
                logger.exception("Model load failed")
                raise

    def ensure_on_cuda(self) -> None:
        """Ensure model is on CUDA. Called inside @spaces.GPU each call."""
        # Force CUDA re-detection (PyTorch caches this at import time)
        try:
            torch.cuda.init()
        except Exception as e:
            logger.warning("torch.cuda.init() failed: {}", e)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available inside @spaces.GPU block. This usually "
                "means ZeroGPU quota is exhausted or the spaces library failed "
                "to attach a GPU."
            )

        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        # Move to CUDA (idempotent if already there)
        self._model = self._model.to("cuda")
        logger.info("Model confirmed on CUDA")

    def infer_multi(self, image_paths: list[str], prompt: str = DEFAULT_PROMPT) -> str:
        """
        Run multi-page OCR. On ZeroGPU this routes through @spaces.GPU.
        Returns the raw model output with <PAGE> markers between pages.
        """
        if self.needs_zero_gpu and _HAS_SPACES:
            return _run_zero_gpu(self, image_paths, prompt, self.settings)

        # Normal env: load lazily, then run.
        self.load()
        return _run_inference(self._model, self._tokenizer, image_paths, prompt, self.settings)


# --------------------------------------------------------------------- #
# Inference kernel (calls model.infer_multi)
# --------------------------------------------------------------------- #
def _run_inference(model, tokenizer, image_paths, prompt, settings) -> str:
    """
    Call model.infer_multi(). Model expects image FILES (paths), not PIL.

    NOTE: model.infer_multi() internally calls .cuda() on its inputs, so
    the model MUST be on CUDA before this is called.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_out:
        outputs, _n_tokens = model.infer_multi(
            tokenizer=tokenizer,
            prompt=prompt,
            image_files=image_paths,
            output_path=tmp_out,
            image_size=1024,  # Multi-page mode only supports 1024
            save_results=False,
            max_length=settings.max_new_tokens,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            temperature=0.0,
        )
    return outputs


# --------------------------------------------------------------------- #
# ZeroGPU wrapper — top-level function (required by @spaces.GPU)
# --------------------------------------------------------------------- #
# `duration=600` gives 10 min per call. Cold start (download + load + infer)
# can take 3-5 min; subsequent calls take 30s-2min.
@spaces.GPU(duration=600)
def _run_zero_gpu(manager: "ModelManager", image_paths, prompt, settings) -> str:
    # 1. Load model (downloads weights if first time, loads to CPU then CUDA)
    manager.load()
    # 2. Ensure on CUDA (forces torch.cuda.init() too)
    manager.ensure_on_cuda()
    # 3. Run inference
    return _run_inference(manager._model, manager._tokenizer, image_paths, prompt, settings)


# --------------------------------------------------------------------- #
@lru_cache
def get_model_manager() -> ModelManager:
    return ModelManager(get_settings())
