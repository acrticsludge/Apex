FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY apex_dashboard.py .

# Gunicorn: 1 worker (single-process so _state/_latest_prices stay in-memory),
# threads for concurrent requests, port from Railway's $PORT env var.
ENV PORT=7000
EXPOSE 7000

CMD ["sh", "-c", "gunicorn apex_dashboard:app \
     --bind 0.0.0.0:${PORT} \
     --workers 1 \
     --threads 4 \
     --timeout 120 \
     --log-level info"]
