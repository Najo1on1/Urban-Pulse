FROM python:3.12-slim

# Install system dependencies required for spatial mapping
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the unified requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy our application code
COPY src/ ./src/
COPY tests/ ./tests/
