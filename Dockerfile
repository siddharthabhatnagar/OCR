FROM python:3.10-slim

# System deps for OpenCV + PaddlePaddle native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Memory optimization: glibc normally holds freed memory in its arena.
# Setting MALLOC_TRIM_THRESHOLD_ to a low value forces it to return freed
# memory to the OS. Critical for fitting PaddleOCR into 512 MB free tier.
ENV MALLOC_TRIM_THRESHOLD_=65536
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build so the first request is fast.
# This downloads ~60 MB of models:
#   - PP-OCRv4 det (table/text detection)
#   - PP-OCRv4 rec (text recognition)
#   - PP-OCRv4 cls (text angle classification)
#   - SLANet table structure model
#
# We don't preload PPStructure here (it needs the layout parser which we
# disabled for memory). PPStructure will download its specific models on
# first request (~30 MB, ~5 s on Render's network).
#
# If the download fails (e.g. network issue), continue — models will
# download on first request instead.
RUN python -c "from paddleocr import PaddleOCR; \
    PaddleOCR(use_angle_cls=True, lang='en', show_log=True)" 2>&1 || \
    echo "WARNING: model pre-download failed, will download on first request"

# Copy app code
COPY app ./app
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

# Render sets PORT env var. We read it in start.sh.
CMD ["bash", "start.sh"]
