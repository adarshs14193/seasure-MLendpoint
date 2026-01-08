# --- Stage 1: Builder ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies (if needed for libraries like numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++

# --- Stage 1: Builder ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++

# Create and use a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Runtime ---
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
# This is crucial for debugging memory issues on Koyeb
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install ONLY the runtime system libraries for OpenCV and ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy your application code
COPY . .

# Run as a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port 8080 (Make sure to set this in Koyeb Network settings)
EXPOSE 8080

# Command to run the application
# Ensure 'api' matches your filename (e.g., api.py) and 'app' is your FastAPI instance
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
