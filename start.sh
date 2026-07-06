#!/bin/bash
set -e

# Render sets PORT automatically. Default to 8000 for local runs.
export PORT=${PORT:-8000}

echo "=================================================="
echo "  PaddleOCR Table API"
echo "  Listening on 0.0.0.0:${PORT}"
echo "  Language: ${PADDLE_LANG:-en}"
echo "  Workers: 1 (PaddleOCR is heavy, don't multi-process)"
echo "=================================================="

# Single worker — PaddleOCR models are ~500 MB in RAM.
# Multiple workers would each load their own copy.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 30 \
    --log-level info
