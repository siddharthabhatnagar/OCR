---
title: Unlimited OCR API
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
short_description: Free OCR API for RAG using baidu/Unlimited-OCR on ZeroGPU.
tags:
  - ocr
  - rag
  - fastapi
  - document-ai
  - unlimited-ocr
models:
  - baidu/Unlimited-OCR
---

# Unlimited-OCR API

Free OCR service for RAG ingestion, powered by **`baidu/Unlimited-OCR`** (3B-param MoE, MIT-licensed) on HuggingFace Spaces ZeroGPU.

## Endpoints

| URL | Method | Purpose |
|---|---|---|
| `/` | GET | Gradio UI (browser file upload) |
| `/health` | GET | Health check (no auth) |
| `/ocr` | POST | OCR a PDF or image (multipart upload) |
| `/ocr/raw` | POST | OCR a raw image (bytes body) |
| `/docs` | GET | Swagger UI |

## Quickstart

### Browser testing
Visit your Space URL → drag a PDF into the upload box → click "Run OCR".

### API (curl)
```bash
curl -X POST https://YOUR-SPACE.hf.space/ocr \
  -F "file=@paper.pdf" \
  -o result.json

python -c "
import json
d = json.load(open('result.json'))
print(f'Pages: {d[\"total_pages\"]}, Time: {d[\"elapsed_ms\"]} ms')
print(d['markdown'][:500])
"
```

### Python client (for RAG)
```python
from client.ocr_client import UnlimitedOcrClient, to_langchain_documents

client = UnlimitedOcrClient(base_url="https://YOUR-SPACE.hf.space")
ocr = client.ocr_pdf("paper.pdf")

# Convert to LangChain Documents (one per page)
docs = to_langchain_documents(ocr, source="paper.pdf")
```

## Configuration (env vars / Space Variables)

| Var | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `baidu/Unlimited-OCR` | HF model repo |
| `MODEL_DTYPE` | `bfloat16` | Model dtype |
| `PDF_DPI` | `144` | Render DPI for PDFs |
| `PDF_MAX_PAGES` | `50` | Hard cap to protect GPU memory |
| `CHUNK_SIZE` | `4` | Pages per `infer_multi()` call (lower = safer for ZeroGPU 5-min limit) |
| `MAX_NEW_TOKENS` | `16384` | Generation cap |
| `API_KEY` | – | If set, clients must send `Authorization: Bearer <key>` |
| `HF_TOKEN` | – | Avoids rate limits on model download |

## Project layout

```
unlimited-ocr-api/
├── app.py                # Gradio entry point (HF Spaces serves this)
├── requirements.txt
├── README.md
├── service/
│   ├── main.py           # FastAPI routes (/ocr, /health, /docs)
│   ├── config.py         # Pydantic settings
│   ├── model.py          # ModelManager: lazy load, ZeroGPU-aware
│   ├── pdf.py            # PyMuPDF rendering
│   ├── ocr.py            # Top-level pipeline (PDF → OcrResponse)
│   ├── postprocess.py    # Markdown → typed Elements
│   └── schemas.py        # Pydantic response models
├── client/
│   └── ocr_client.py     # Sync/async client, LangChain/LlamaIndex adapters
└── tests/
    └── test_api.py       # Smoke tests
```

## Cold start

First OCR request after idle takes **2-5 minutes** (download 6 GB model + load to GPU). Subsequent calls take 30s-2min depending on PDF size. ZeroGPU quota: ~80 GPU-min/day on free tier.

## License

Code: MIT. Model weights: see [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR).
