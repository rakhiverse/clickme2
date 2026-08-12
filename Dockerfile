FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ANONYMIZED_TELEMETRY=False

# Install system dependencies required for OpenCV
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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads clickme_db && chmod -R 777 uploads clickme_db /app

# Pre-download detection and recognition modules during build
RUN python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition']); app.prepare(ctx_id=-1, det_size=(640, 640))" || true

EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
