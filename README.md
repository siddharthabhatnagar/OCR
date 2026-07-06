# Table OCR — Vercel-deployable

Extract tables from images using Microsoft's **Table Transformer (TATR)** for
table detection and structure recognition, plus **Tesseract.js** for cell-level
OCR. All inference runs inside the Vercel serverless function — no external
API calls, no GPU, no Pinecone, no nothing.

> **One-shot deploy**: `unzip table-ocr-vercel.zip && cd table-ocr-vercel && npm install && vercel`

---

## Size budget — fits under Vercel's 250 MB limit

| Component                                          | Size     |
| -------------------------------------------------- | -------- |
| TATR detection (uint8 ONNX)                        | ~30 MB   |
| TATR structure recognition (uint8 ONNX)            | ~30 MB   |
| Tesseract.js worker + `eng.traineddata`            | ~15 MB   |
| `onnxruntime-node` (bundled w/ transformers.js)    | ~20 MB   |
| `sharp` native binary                              | ~10 MB   |
| Next.js runtime + app code                         | ~30 MB   |
| **Total uncompressed serverless function**         | **~135 MB ✅** |

Models are **not** bundled — they are downloaded to `/tmp` on the first cold
start, then cached for the lifetime of the warm serverless instance.

---

## Deploy on Vercel

### Option A — Vercel CLI (fastest)

```bash
npm i -g vercel
unzip table-ocr-vercel.zip
cd table-ocr-vercel
npm install
vercel        # follow prompts; accept all defaults
vercel --prod  # promote to production
```

### Option B — GitHub import

1. Push the unzipped folder to a new GitHub repo.
2. Vercel dashboard → **New Project** → import the repo.
3. Framework preset: **Next.js** (auto-detected).
4. No env vars required. Click **Deploy**.

That's it. Vercel will read `vercel.json` (60 s timeout, 1 GB RAM) and
`next.config.mjs` (native packages externalized).

---

## Local dev

```bash
npm install
npm run dev
# open http://localhost:3000
```

First request triggers a model download (~75 MB total) into your local HF
cache. Subsequent requests are fast.

---

## Usage

1. Open the deployed URL.
2. Drag a PNG/JPEG/WebP image of a table onto the upload zone (max 4 MB).
3. Click **Extract Table**.
4. See the rendered HTML table + raw JSON cells.

First request after a cold start takes ~10–20 s (model download + worker
init). Subsequent requests on the warm instance take 1–5 s, depending on
table size.

---

## API

### `POST /api/extract-table`

**Body**: `multipart/form-data` with a field `file` containing the image.

**Response** (200):
```json
{
  "rows": 5,
  "cols": 3,
  "cells": [
    { "text": "Name",   "row": 0, "col": 0 },
    { "text": "Age",    "row": 0, "col": 1 },
    { "text": "City",   "row": 0, "col": 2 },
    { "text": "Alice",  "row": 1, "col": 0 }
  ],
  "html": "<table><tr><th>Name</th>…</tr></table>",
  "elapsed_ms": 4321,
  "table_bbox": { "xmin": 12, "ymin": 34, "xmax": 980, "ymax": 760 }
}
```

**Errors** (400 / 413 / 500): `{ "error": "human-readable message" }`

### `GET /api/extract-table`
Returns endpoint metadata — useful for health checks.

---

## Configuration (optional env vars)

| Variable               | Default                                                            | Description                              |
| ---------------------- | ------------------------------------------------------------------ | ---------------------------------------- |
| `DETECTION_MODEL_ID`   | `Xenova/table-transformer-detection`                               | HF repo for the detection stage          |
| `STRUCTURE_MODEL_ID`   | `Xenova/table-transformer-structure-recognition-v1.1-all`          | HF repo for the structure stage          |
| `TESSERACT_LANG`       | `eng`                                                              | Tesseract language code (e.g. `eng+fra`) |

Set these in `vercel env` or the project dashboard.

---

## How it works

```
image bytes
    │
    ▼
[sharp] strip alpha → raw RGB
    │
    ▼
[transformers.js] TATR detection (object-detection)
    │  → table bbox
    ▼
[sharp] crop to table bbox
    │
    ▼
[transformers.js] TATR structure recognition (object-detection)
    │  → rows, columns, header rows (sorted)
    ▼
for each (row, col) cell:
    [sharp] extract cell PNG
    [tesseract.js] OCR the cell → text
    │
    ▼
build cells[] + HTML table
    │
    ▼
JSON response
```

---

## Limitations & gotchas

- **4 MB upload cap** — Vercel serverless body limit is 4.5 MB. For larger
  images, downscale first or upload via [Vercel Blob](https://vercel.com/docs/storage/vercel-blob)
  and POST the URL.
- **60 s timeout** on Hobby plan. Complex tables with many cells may exceed
  this — upgrade to Pro for 300 s.
- **CPU-only** — no GPU on Vercel serverless. Inference is ~5–20× slower than
  a GPU box. A 5×5 table typically takes 2–4 s on a warm instance.
- **Cold start** — first request after deployment (or after the instance goes
  idle, ~5 min on Hobby) downloads ~75 MB of weights to `/tmp`. Be patient.
- **Header detection** — TATR tags the topmost row(s) as `row header`. The
  pipeline renders these as `<th>`. If your table has no header, the first
  row is still rendered as `<th>` — harmless.
- **Spanning cells** — TATR emits `spanning cell` labels for merged cells;
  the current pipeline does **not** reconstruct `rowspan`/`colspan`. PRs
  welcome.
- **Non-English text** — set `TESSERACT_LANG` to e.g. `fra`, `deu`, `chi_sim`
  etc. Multiple languages: `eng+fra`.

---

## Models used

| Stage    | Original (PyTorch)                                                   | ONNX export (used here)                                                |
| -------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Detect   | [`microsoft/table-transformer-detection`](https://huggingface.co/microsoft/table-transformer-detection) | [`Xenova/table-transformer-detection`](https://huggingface.co/Xenova/table-transformer-detection) |
| Structure | [`microsoft/table-transformer-structure-recognition-v1.1-all`](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all) | [`Xenova/table-transformer-structure-recognition-v1.1-all`](https://huggingface.co/Xenova/table-transformer-structure-recognition-v1.1-all) |
| OCR      | —                                                                    | `tesseract.js` (LSTM, English)                                          |

All three run inside the same serverless function via `onnxruntime-node`
(bundled with `@huggingface/transformers` v3) and the Tesseract WASM core.

---

## File layout

```
table-ocr-vercel/
├── package.json
├── next.config.mjs          # externalizes native deps
├── vercel.json              # 60 s timeout, 1 GB RAM
├── tsconfig.json
├── .env.example             # optional overrides
├── .gitignore
├── README.md                # this file
├── app/
│   ├── layout.tsx
│   ├── page.tsx             # upload UI + result rendering
│   ├── globals.css
│   └── api/
│       └── extract-table/
│           └── route.ts     # POST handler
└── lib/
    ├── models.ts            # lazy model singletons + OCR worker
    └── pipeline.ts          # detect → crop → structure → cell OCR → HTML
```

---

## License

MIT for the wrapper code in this repo.

Model weights follow their original licenses:
- Microsoft Table Transformer — MIT
- Xenova ONNX exports — MIT
- Tesseract LSTM data — Apache-2.0

---

## Troubleshooting

**`Error: ... onnxruntime-node ... native module not found`**
Make sure `next.config.mjs` lists `onnxruntime-node` in
`serverExternalPackages`. It is — re-run `npm install` if you edited
`package.json`.

**First deploy fails with `250 MB exceeded`**
Run `vercel build --debug` locally and check the bundle. The most common
culprit is accidentally importing a server-side module in client code.

**Cold start is slow (30 s+)**
That's the model download. After the first request, the warm instance caches
weights in `/tmp` and subsequent requests are fast. To keep the instance
warm, ping `/api/extract-table` (GET) every 4 minutes with a cron job
(e.g. Upstash QStash, GitHub Actions, cron-job.org).

**OCR returns garbage for non-Latin text**
Set `TESSERACT_LANG=chi_sim` (or `jpn`, `kor`, `ara`, etc.) and redeploy.

**Table is detected but cells are empty**
The structure recognition likely failed. Try a higher-resolution, higher-
contrast image. TATR struggles with screenshots of dark-themed UIs.
