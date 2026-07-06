# Docling OCR API — Vercel Deployment

Free OCR service for RAG ingestion, powered by **IBM Docling** on Vercel's free Hobby tier. Pure CPU — no GPU, no quota, no cold-start hell.

## Why Docling + Vercel?

| Trait | Value |
|---|---|
| **Cost** | $0 forever (Vercel Hobby + CPU only) |
| **Quota** | 100 GB bandwidth/month, 100 GB-hr function execution |
| **Cold start** | ~5 sec |
| **Per-page OCR** | 2-5 sec |
| **Tables** | ✅ Excellent (TableFormer model) |
| **Equations** | ⚠️ Basic |
| **Diagrams** | ⚠️ Caption only |
| **Bundle size** | ~220 MB (fits in Vercel's 250 MB limit) |
| **GPU required** | ❌ No |

## Endpoints

| URL | Method | Purpose |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/ocr` | POST | OCR a PDF or image (multipart upload) |
| `/docs` | GET | Swagger UI |
| `/openapi.json` | GET | OpenAPI spec |

## Deploy to Vercel (5 steps)

### Step 1 — Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2 — Login

```bash
vercel login
```

### Step 3 — Deploy from project directory

```bash
cd docling-ocr-vercel
vercel --prod
```

- When prompted for project settings, accept defaults.
- Vercel auto-detects Python from `requirements.txt`.
- First deploy takes ~3-5 min (installs Docling + torch CPU).

### Step 4 — Note your URL

After deploy, Vercel prints:
```
Production: https://docling-ocr-vercel-xxx.vercel.app
```

### Step 5 — Verify

```bash
URL="https://your-app.vercel.app"

# Health
curl $URL/health

# Browser docs
open $URL/docs

# OCR a PDF
curl -X POST $URL/ocr -F "file=@paper.pdf" -o result.json
python -c "
import json
d = json.load(open('result.json'))
print(f'Pages: {d[\"total_pages\"]}, Time: {d[\"elapsed_ms\"]} ms')
print(d['markdown'][:500])
"
```

## Use the Python client (for RAG)

```python
from client.ocr_client import OcrClient, to_langchain_documents

client = OcrClient(base_url="https://your-app.vercel.app")
ocr = client.ocr_pdf("paper.pdf")

print(ocr.markdown)            # full document markdown
print(ocr.total_pages)         # page count
print(ocr.elapsed_ms)          # processing time

# Per-page typed elements (text/heading/table/equation/figure)
for page in ocr.pages:
    for el in page.elements:
        print(el.type, el.content[:80])

# Convert to LangChain Documents
docs = to_langchain_documents(ocr, source="paper.pdf")
```

## Project layout

```
docling-ocr-vercel/
├── api/                    # Vercel serverless functions (one per route)
│   ├── _serverless.py      # Shared helpers (CORS, JSON response)
│   ├── index.py            # GET / — service info
│   ├── health.py           # GET /health
│   ├── ocr.py              # POST /ocr — multipart upload + OCR
│   ├── docs.py             # GET /docs — Swagger UI
│   └── openapi.py          # GET /openapi.json
├── service/                # Business logic
│   ├── ocr.py              # Docling OCR pipeline
│   └── schemas.py          # Pydantic response models
├── client/
│   └── ocr_client.py       # Python client + LangChain/LlamaIndex helpers
├── tests/
│   └── test_api.py         # Integration smoke tests
├── requirements.txt
├── vercel.json             # Vercel config (routes, memory, timeout)
├── package.json
├── .gitignore
├── .env.example
└── README.md
```

## API response shape

```jsonc
{
  "markdown": "# Title\n\nFull document markdown...",
  "total_pages": 3,
  "elapsed_ms": 4823,
  "model": "docling",
  "warnings": [],
  "pages": [
    {
      "page": {"page_index": 0, "width_px": 612, "height_px": 792, "dpi": 72},
      "markdown": "# Title\n\nPage 1 markdown...",
      "elements": [
        {"type": "heading", "content": "Title", "page_index": 0, "extra": {}},
        {"type": "table",  "content": "| A | B |\n|---|---|\n| 1 | 2 |", "page_index": 0},
        {"type": "text",   "content": "Body paragraph ...", "page_index": 0}
      ]
    }
  ]
}
```

## Local development

```bash
# Install deps
pip install -r requirements.txt

# Run Vercel dev server (needs Vercel CLI)
npm install -g vercel
vercel dev

# Or run as plain Python (each endpoint is a standard http.server handler)
python -m http.server 3000
```

## Operational notes

| Concern | Recommendation |
|---|---|
| **Function timeout** | 60s default (set in `vercel.json`). Raise if OCR-ing large PDFs. |
| **Memory** | 1024 MB default. Docling + a 50-page PDF uses ~700 MB. |
| **Cold start** | ~5 sec. Vercel may spin down idle functions. Schedule a cron `/health` ping. |
| **Bundle size** | Docling + torch CPU = ~220 MB. Vercel limit is 250 MB. If you hit it, switch to PyMuPDF4LLM (Option 2). |
| **Free tier limits** | 100 GB bandwidth/month, 100 GB-hr function execution. Plenty for `<100 docs/day`. |
| **Caching** | Same upload = same result. Consider adding a Redis cache (Upstash free tier) if you OCR the same docs repeatedly. |

## When this is NOT enough

Switch to GPU options if you need:
- **LaTeX equation rendering** → use Unlimited-OCR (Modal, $30/mo free credits)
- **Diagram understanding** → use a VLM (GLM-4.5V, free via z-ai SDK)
- **Handwriting recognition** → use Unlimited-OCR or TrOCR
- **Scanned PDFs at scale** → use PaddleOCR (free, can run CPU-only)

## Comparison with Unlimited-OCR

| Feature | Unlimited-OCR (ZeroGPU) | Docling (Vercel) |
|---|---|---|
| Deploy cost | Free (quota-limited) | Free (unlimited) |
| Cold start | 2-5 min | ~5 sec |
| Per-page time | 20-40 sec | 2-5 sec |
| Table accuracy | Excellent | Very good |
| Equation (LaTeX) | ✅ Excellent | ⚠️ Basic |
| Diagram understanding | ✅ Caption + bbox | ⚠️ Caption only |
| Handwriting | ✅ Good | ❌ |
| Multi-language | ✅ 70+ langs | ✅ 80+ langs |
| Quota | ~90 sec GPU/day | 100 GB-hr/month |

## License

Code: MIT. Docling: MIT (IBM).
