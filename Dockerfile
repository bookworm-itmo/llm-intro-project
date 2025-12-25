# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

WORKDIR /app

# Install build tools (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install Python deps with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy only necessary code (not data!)
COPY services/ ./services/
COPY frontend/ ./frontend/
COPY main.py .

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
