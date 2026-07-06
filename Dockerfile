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
#   - Layout parser model
#
# If the download fails (e.g. network issue), continue — models will
# download on first request instead.
RUN python -c "from paddleocr import PaddleOCR, PPStructure; \
    PaddleOCR(use_angle_cls=True, lang='en', show_log=True); \
    PPStructure(show_log=True, lang='en')" 2>&1 || \
    echo "WARNING: model pre-download failed, will download on first request"

# Copy app code
COPY app ./app
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000

# Render sets PORT env var. We read it in start.sh.
CMD ["bash", "start.sh"]
