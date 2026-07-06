# Deploy to HuggingFace Spaces (FREE, ZeroGPU)

This is the canonical free deployment path. **ZeroGPU only works with the Gradio SDK** (not Docker SDK).

## Steps

### 1. Create the Space

1. Go to <https://huggingface.co/new-space>
2. **Owner**: your username
3. **Space name**: `unlimited-ocr` (lowercase, no underscores)
4. **License**: MIT
5. **SDK**: **Gradio** ← critical for ZeroGPU
6. **Hardware**: **ZeroGPU** ← free A100 shared
7. **Visibility**: Public (required for free tier)
8. Click **Create Space**

### 2. Set secrets/variables (optional)

Space → **Settings** → **Variables and secrets**:

| Name | Type | Purpose |
|---|---|---|
| `HF_TOKEN` | Secret | Avoids rate limits on model download |
| `API_KEY` | Secret | If set, clients must send `Authorization: Bearer <key>` |
| `CHUNK_SIZE` | Variable | Pages per OCR call (default 4; lower = safer for ZeroGPU timeout) |
| `MAX_NEW_TOKENS` | Variable | Generation cap (default 16384) |

### 3. Push the code

```bash
# Unzip the project
unzip unlimited-ocr-api.zip
cd unlimited-ocr-api

# Init git and add HF remote
git init
git add .
git commit -m "init: Unlimited-OCR FastAPI + Gradio UI on ZeroGPU"
git remote add space https://huggingface.co/spaces/<your-username>/unlimited-ocr

# Push (use HF username + token as password)
git push -u space main
```

### 4. Wait for build (~5-10 min)

Watch the **Logs** tab in your Space. Build downloads ~6 GB of dependencies.

### 5. Verify

```bash
# Health (should return ok, zero_gpu not in schema but hf_space=true)
curl https://<your-username>-unlimited-ocr.hf.space/health

# Browser UI
open https://<your-username>-unlimited-ocr.hf.space/

# First OCR call (cold start, 2-5 min)
curl -X POST https://<your-username>-unlimited-ocr.hf.space/ocr \
  -F "file=@paper.pdf" \
  --max-time 600 \
  -o result.json

python -c "
import json
d = json.load(open('result.json'))
print(f'Pages: {d[\"total_pages\"]}, Time: {d[\"elapsed_ms\"]} ms')
print(d['markdown'][:500])
"
```

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `Cannot install spaces==A and spaces==B` | Pinned spaces in requirements.txt | Remove spaces, gradio, torch, uvicorn from requirements (HF pre-installs them) |
| `Attribute "app" not found in module "app"` | Name collision: `app/` package vs `app.py` file | Rename package to `service/` (already done in this project) |
| `address already in use` | Double server startup (`__main__` block + SDK launch) | Use `demo.launch(prevent_reloop=True)` in `__main__` (already done) |
| `Found no NVIDIA driver` | ZeroGPU not active | Ensure SDK is Gradio (not Docker) and Hardware is ZeroGPU |
| `CUDA not available inside @spaces.GPU` | Stale torch CUDA cache | Call `torch.cuda.init()` inside @spaces.GPU (already done) |
| `Unrecognized configuration class ... AutoModelForCausalLM` | Wrong AutoModel class | Use `AutoModel` (the model's config.json registers under AutoModel) |
| First OCR call times out | ZeroGPU 5-min limit, model loading too slow | Lower `CHUNK_SIZE=2` and `MAX_NEW_TOKENS=8192` |

## Cold start behavior

| Event | Duration |
|---|---|
| Container startup | ~5 sec |
| First OCR call: model download | 1-3 min (6 GB) |
| First OCR call: model load to GPU | ~30 sec |
| First OCR call: inference | ~10-30 sec |
| Subsequent calls (model cached) | 30 sec - 2 min |

ZeroGPU quota: ~80 GPU-min/day on free tier. Schedule a 5-min cron `/health` ping to keep the Space warm.

## Migrating off free tier

When you outgrow ZeroGPU quota:
1. **Paid HF Space** — flip Hardware to a paid GPU (A10G ~$0.60/hr). Same code, no changes.
2. **Modal** — copy the project, decorate OCR handler with `@modal.function(gpu="A10G")`.
3. **RunPod Serverless** — push the Dockerfile, point worker at port 7860.
