# PaddleOCR Table API — Deploy on Render

A FastAPI service that runs **PaddleOCR's PP-StructureV2** to extract tables from images. Returns HTML + structured cells (with rowspan/colspan support).

Designed as a **drop-in backend** for the Vercel Next.js frontend — same API contract, same response shape.

---

## Why Render instead of Vercel?

| | Vercel (Next.js) | Render (FastAPI + PaddleOCR) |
|---|---|---|
| Max function size | 250 MB | No limit (Docker) |
| Native C++ libs | ❌ Hard | ✅ Easy (Docker) |
| PaddlePaddle | ❌ Doesn't fit | ✅ ~150 MB, no problem |
| Table OCR quality | ⭐⭐⭐ (TATR + Tesseract) | ⭐⭐⭐⭐⭐ (PP-StructureV2) |
| Cold start | 10–20 s | 5–10 s (models pre-baked) |
| Cost | Free tier OK | Starter $7/mo recommended |

---

## Quick deploy

### Option A — Render Blueprint (one-click)

1. Push this folder to a new GitHub repo.
2. Go to https://dashboard.render.com/blueprints
3. Click **New Blueprint Instance** → select your repo.
4. Render reads `render.yaml` and deploys automatically.
5. Wait ~5 min for Docker build + model pre-download.
6. Your API is live at `https://paddle-ocr-table-XXXX.onrender.com`.

### Option B — Manual (Render dashboard)

1. Push this folder to GitHub.
2. Render dashboard → **New +** → **Web Service**.
3. Connect your repo.
4. Settings:
   - **Runtime**: Docker
   - **Plan**: Starter ($7/mo) or Standard ($25/mo)
   - **Dockerfile Path**: `./Dockerfile`
   - **Health Check Path**: `/health`
5. Add env vars (optional):
   - `PADDLE_LANG` = `en` (or `ch`, `french`, `german`, `korean`, `japan`, ...)
   - `CORS_ORIGINS` = `https://your-vercel-app.vercel.app` (restrict access)
6. Click **Create Web Service**. Wait ~5 min.

### Option C — Local Docker (for testing)

```bash
cd paddle-ocr-render
docker build -t paddle-ocr-table .
docker run -p 8000:8000 paddle-ocr-table

# Test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/extract-table \
  -F "file=@some-table.png" | python -m json.tool
```

---

## API endpoints

### `GET /health`
```json
{ "status": "ok", "service": "paddle-ocr-table", "models_loaded": true, "language": "en" }
```

### `GET /`
Returns endpoint listing.

### `GET /docs`
Interactive Swagger UI — upload images and test directly in the browser.

### `POST /ocr`
Plain text OCR. Returns every text line with bounding box + confidence.

**Request**: `multipart/form-data` with `file` field (PNG/JPEG/WebP/BMP/TIFF, max 10 MB)

**Response**:
```json
{
  "items": [
    {
      "text": "Hello World",
      "confidence": 0.98,
      "bbox": { "xmin": 10, "ymin": 20, "xmax": 200, "ymax": 50 }
    }
  ],
  "count": 1,
  "elapsed_ms": 120
}
```

### `POST /table`
Full table OCR. Returns all tables found in the image.

**Response**:
```json
{
  "tables": [
    {
      "bbox": [10, 20, 980, 760],
      "html": "<html><body><table>...</table></body></html>",
      "cell_bbox": [[...], [...]]
    }
  ],
  "count": 1,
  "elapsed_ms": 850
}
```

### `POST /api/extract-table` ← Vercel drop-in

Same response shape as the original Vercel Next.js endpoint. **This is the one your frontend calls.**

**Request**: `multipart/form-data` with `file` field

**Response** (200):
```json
{
  "rows": 5,
  "cols": 3,
  "cells": [
    { "text": "Name",   "row": 0, "col": 0, "rowspan": 1, "colspan": 1 },
    { "text": "Age",    "row": 0, "col": 1, "rowspan": 1, "colspan": 1 },
    { "text": "City",   "row": 0, "col": 2, "rowspan": 1, "colspan": 1 },
    { "text": "Alice",  "row": 1, "col": 0, "rowspan": 1, "colspan": 1 }
  ],
  "html": "<table><tr><td>Name</td>...</tr></table>",
  "elapsed_ms": 850,
  "table_bbox": { "xmin": 10, "ymin": 20, "xmax": 980, "ymax": 760 }
}
```

**Errors**:
- `400` — unsupported file type
- `413` — file > 10 MB
- `422` — no table detected in image
- `500` — internal error (check Render logs)

---

## Wire it up to your Vercel frontend

In your Next.js project, edit `app/page.tsx`:

```tsx
// Change this line:
const res = await fetch('/api/extract-table', {

// To this:
const OCR_API_URL = process.env.NEXT_PUBLIC_OCR_API_URL || '/api/extract-table';
const res = await fetch(OCR_API_URL, {
  method: 'POST',
  body: form,
});
```

Then set the env var on Vercel:
```bash
vercel env add NEXT_PUBLIC_OCR_API_URL production
# Paste: https://paddle-ocr-table-XXXX.onrender.com
vercel --prod
```

You can also **delete** the Vercel API route (`app/api/extract-table/route.ts`) and the `lib/` folder — Vercel now just hosts the static UI, Render does the heavy lifting.

---

## How PaddleOCR's table pipeline works

```
image
  │
  ▼
[Layout parser]  →  detects regions (table, figure, title, text)
  │
  ▼ table region crop
  │
[SLANet table structure]  →  generates HTML with <tr>/<td> + colspan/rowspan
  │
  ▼
[PP-OCRv4]  →  recognizes text inside each cell
  │
  ▼
HTML table with text content
  │
  ▼
[BeautifulSoup parser]  →  structured cells[] with row/col mapping
  │
  ▼
JSON response
```

This is fundamentally different (and better) than the TATR + Tesseract.js approach:

| Aspect | TATR + Tesseract | PaddleOCR PP-Structure |
|---|---|---|
| Table structure model | Detects rows/cols separately | Single end-to-end model |
| Merged cells | ❌ Not supported | ✅ colspan/rowspan |
| Cell OCR | Tesseract (per-cell, slow) | PP-OCRv4 (batch, fast) |
| Skewed tables | ❌ Fails | ✅ Handles rotation |
| Languages | Tesseract packs (~5 MB each) | 80+ PaddleOCR langs |
| Speed (5×5 table) | ~5 s | ~1 s |

---

## Resource usage

| Render plan | RAM | CPU | Price | Can run PaddleOCR? |
|---|---|---|---|---|
| Free | 512 MB | 0.1 | $0 | ⚠️ Borderline — see below |
| Starter | 512 MB | 0.1 | $7/mo | ⚠️ Same memory as free, just no sleep |
| Standard | 2 GB | 0.5 | $25/mo | ✅ Recommended (~1 s/table) |
| Pro | 4 GB | 1.0 | $50/mo | ✅ Fast (~0.5 s/table) |

### Free tier — honest expectations

PaddleOCR + PaddlePaddle at runtime uses **~450-550 MB RAM**. Render's free tier is **512 MB**. This is borderline. Here's what to expect:

| Image type | Will it work? | Notes |
|---|---|---|
| Small (<500 KB), simple table (5×5) | ✅ Yes | ~3-5 s/response |
| Medium (1-2 MB), complex table (10×10) | ⚠️ Maybe | Right at the limit, may OOM |
| Large (>2 MB), multi-table page | ❌ No | Will OOM-kill |
| First request after cold start | ❌ Likely OOM | Loading all models at once spikes memory |

**Memory optimizations already applied** in this build:
- `layout=False` — skips the heavy layout parser (~50 MB saved)
- `MALLOC_TRIM_THRESHOLD_=65536` — glibc returns freed memory to OS
- `gc.collect()` after every request — Python frees image buffers
- Lazy model loading — only loads what each endpoint needs
- Single uvicorn worker — no duplicate model copies

**If you still hit OOM**, options in order of preference:
1. **Downscale images before upload** (e.g. resize to max 1500px wide in your Vercel frontend with `sharp`)
2. **Upgrade to Standard ($25/mo)** — 2 GB RAM, fits comfortably
3. **Switch to RapidOCR** (ONNX-based PaddleOCR port, ~150 MB runtime) — fits free tier easily but loses the SLANet table structure model

### Cold start on free tier

Render free tier sleeps after 15 min of inactivity. Cold starts take **~60-90 seconds**:
- Container spins up (~5 s)
- Python imports load (~10 s)
- PaddlePaddle framework loads (~20 s)
- Models load from disk to RAM (~15 s)
- First inference warms up MKL-DNN (~10 s)

To avoid cold starts during active use, ping `/health` every 10 minutes with an external cron (e.g. cron-job.org, UptimeRobot, GitHub Actions).

Memory breakdown at runtime (after optimizations):
- PaddlePaddle runtime: ~200 MB
- PaddleOCR models (PP-OCRv4 + SLANet): ~250 MB
- Image buffers + processing: ~50-100 MB
- **Total: ~500-550 MB** → fits free tier barely

---

## Configuration

All via env vars on Render:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Set automatically by Render |
| `PADDLE_LANG` | `en` | OCR language. Options: `en`, `ch`, `french`, `german`, `korean`, `japan`, `arabic`, `hindi`, `italian`, `portuguese`, `russian`, `spanish`, ... |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. Set to your Vercel URL in production. |

Set them in Render dashboard → **Environment** tab, or in `render.yaml`.

---

## Local dev (without Docker)

```bash
cd paddle-ocr-render
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

First run downloads ~60 MB of models to `~/.paddleocr/`.

---

## Testing

```bash
# Health check
curl https://your-app.onrender.com/health

# Plain OCR
curl -X POST https://your-app.onrender.com/ocr \
  -F "file=@receipt.png" | jq .

# Table OCR (Vercel-compatible)
curl -X POST https://your-app.onrender.com/api/extract-table \
  -F "file=@table.png" | jq .

# Or just open the Swagger UI in your browser:
open https://your-app.onrender.com/docs
```

---

## File structure

```
paddle-ocr-render/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI endpoints
│   ├── ocr.py             # PaddleOCR + PPStructure wrappers (lazy singletons)
│   └── table_parser.py    # HTML → structured cells (handles colspan/rowspan)
├── Dockerfile             # Python 3.10 + system deps + model pre-download
├── requirements.txt
├── render.yaml            # Render Blueprint (one-click deploy)
├── start.sh               # uvicorn launcher
├── .dockerignore
└── README.md              # this file
```

---

## Troubleshooting

**Build fails with `ModuleNotFoundError: No module named 'paddle'`**
PaddlePaddle wheel is platform-specific. The Dockerfile uses `python:3.10-slim` on Linux x86_64, which has pre-built wheels. If you're on Apple Silicon locally, use `--platform linux/amd64` when building Docker.

**First request is slow (30+ seconds)**
Models are downloading. The Dockerfile pre-downloads them during build, but if that step failed (network issue), they download on first request. Check the build logs for "WARNING: model pre-download failed".

**`OOMKilled` or 502 errors on large images**
512 MB RAM is tight. Either:
- Upgrade to Standard plan (2 GB)
- Downscale images before uploading (`sharp` in the Vercel frontend, or `Pillow` here)

**CORS error in browser console**
Set `CORS_ORIGINS=https://your-vercel-app.vercel.app` on Render. Or leave as `*` for prototyping.

**OCR returns Chinese text as garbage**
Set `PADDLE_LANG=ch` on Render. For mixed Chinese+English, use `PADDLE_LANG=ch` (the `ch` model includes English).

**Table HTML is empty**
PP-Structure didn't detect a table. Try:
- Higher resolution image (at least 500px wide for the table)
- Better contrast (dark text on white background)
- Crop to just the table region

**Render deployment is slow (~5 min)**
Docker build installs paddlepaddle (~150 MB) + pre-downloads models (~60 MB). First build is slow, subsequent builds use Docker layer cache and are faster.

---

## Cost summary

| Component | Monthly cost |
|---|---|
| Vercel (frontend + static) | $0 (Hobby) |
| Render Starter (512 MB) | $7 |
| Render Standard (2 GB, recommended) | $25 |
| Domain (optional) | $10/yr |

**Total: $7–25/mo** for a production-ready table OCR service.

---

## License

MIT for the wrapper code. PaddleOCR and PaddlePaddle are Apache-2.0.
