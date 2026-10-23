FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

# Hugging Face Spaces runs containers as a non-root user (uid 1000)
RUN useradd -m -u 1000 user

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .
RUN chown -R user:user /app

USER user

# Build the FAISS index (and cache the embedding model) at image build time
# so the container starts fast and never builds the index on first request.
RUN python -m app.ingest

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
