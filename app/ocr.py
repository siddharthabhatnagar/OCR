"""PaddleOCR wrappers — lazy singletons + extraction functions."""
import os
import logging
import threading
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Language for OCR. Set PADDLE_LANG env var to override (e.g. 'ch', 'french').
# PaddleOCR supports: en, ch, french, german, korean, japan, arabic, hindi, ...
LANG = os.environ.get("PADDLE_LANG", "en")

_ocr_instance: Optional[Any] = None
_table_instance: Optional[Any] = None
_init_lock = threading.Lock()


def get_ocr():
    """Lazy PaddleOCR singleton — loads PP-OCRv4 det + rec + cls models."""
    global _ocr_instance
    if _ocr_instance is None:
        with _init_lock:
            if _ocr_instance is None:
                logger.info("Initializing PaddleOCR (lang=%s)...", LANG)
                from paddleocr import PaddleOCR
                _ocr_instance = PaddleOCR(
                    use_angle_cls=True,
                    lang=LANG,
                    show_log=False,
                    use_gpu=False,
                    enable_mkldnn=True,
                )
                logger.info("PaddleOCR ready")
    return _ocr_instance


def get_table_engine():
    """Lazy PPStructure singleton — loads layout + SLANet table structure models."""
    global _table_instance
    if _table_instance is None:
        with _init_lock:
            if _table_instance is None:
                logger.info("Initializing PPStructure (lang=%s)...", LANG)
                from paddleocr import PPStructure
                _table_instance = PPStructure(
                    show_log=False,
                    use_gpu=False,
                    enable_mkldnn=True,
                    layout=True,
                    table=True,
                    ocr=True,
                    structure_version="v2",
                    lang=LANG,
                )
                logger.info("PPStructure ready")
    return _table_instance


def extract_text(img: np.ndarray) -> List[Dict[str, Any]]:
    """Run plain OCR on an image, return list of text items with bboxes."""
    ocr = get_ocr()
    result = ocr.ocr(img, cls=True)
    items: List[Dict[str, Any]] = []
    if result and result[0]:
        for line in result[0]:
            try:
                box = line[0]
                text, conf = line[1]
            except (IndexError, TypeError, ValueError):
                continue
            items.append({
                "text": str(text),
                "confidence": float(conf),
                "bbox": {
                    "xmin": float(box[0][0]),
                    "ymin": float(box[0][1]),
                    "xmax": float(box[2][0]),
                    "ymax": float(box[2][1]),
                },
            })
    return items


def extract_tables(img: np.ndarray) -> List[Dict[str, Any]]:
    """Run PP-Structure table extraction. Returns list of detected tables."""
    engine = get_table_engine()
    result = engine(img)
    tables: List[Dict[str, Any]] = []
    if not result:
        return tables
    for region in result:
        if not isinstance(region, dict):
            continue
        if region.get("type") != "table":
            continue
        res = region.get("res") or {}
        html = res.get("html", "") or ""
        cell_bbox = res.get("cell_bbox", []) or []
        tables.append({
            "bbox": region.get("bbox"),
            "html": html,
            "cell_bbox": cell_bbox,
        })
    return tables


def decode_image(content: bytes) -> np.ndarray:
    """Decode image bytes (PNG/JPEG/WebP/BMP) to a BGR numpy array."""
    import cv2
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Failed to decode image — not a valid PNG/JPEG/WebP/BMP file"
        )
    return img
