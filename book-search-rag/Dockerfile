# Build arguments
ARG PYTHON_VERSION=3.10-slim

FROM python:${PYTHON_VERSION} as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data folders
COPY src/ src/
COPY main.py .

# Create data directories
RUN mkdir -p data/raw data/processed

# Default environment variables
ENV PORT=5000
ENV FLASK_APP=src/api/server.py

# Expose port
EXPOSE 5000

# Entry point: run the API server
CMD ["python", "src/api/server.py"]
