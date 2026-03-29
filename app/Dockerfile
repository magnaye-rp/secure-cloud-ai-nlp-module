FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY training_data.json* ./
COPY templates/ ./templates/
COPY static*    ./static/

RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    DATA_PATH=/app/training_data.json \
    CONFIDENCE_THRESHOLD=0.35 \
    RATE_LIMIT="60/minute" \
    PORT=8000

EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "30"]