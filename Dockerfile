FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False

# Keep CPU usage controlled
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# System dependencies required by OpenCV / InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create application directories
RUN mkdir -p uploads clickme_db \
    && chmod -R 777 uploads clickme_db /app

# ============================================================
# PRE-DOWNLOAD THE SAME MODEL USED BY face_utils.py
# ============================================================

RUN python -c "\
import insightface; \
app = insightface.app.FaceAnalysis( \
    name='buffalo_s', \
    allowed_modules=['detection', 'recognition'] \
); \
app.prepare(ctx_id=-1, det_size=(320, 320))"

# Render provides PORT automatically
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]