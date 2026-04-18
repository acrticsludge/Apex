FROM python:3.12-slim

WORKDIR /app

# Install deps first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Make trading_agent importable as a package
RUN pip install --no-deps -e .

ENV PORT=7000
EXPOSE 7000

CMD ["sh", "-c", "gunicorn apex_dashboard:app \
     --bind 0.0.0.0:${PORT} \
     --workers 1 \
     --threads 4 \
     --timeout 120 \
     --log-level info"]
