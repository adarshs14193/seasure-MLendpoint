# --- Stage 1: Builder ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies (if needed for libraries like numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++

# Create and use a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Runtime ---
FROM python:3.10-slim

WORKDIR /app

# Install ONLY the runtime system libraries (the fix from our first step)
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

# Ensure the app uses the virtual environment's python
ENV PATH="/opt/venv/bin:$PATH"

# Copy your application code
COPY . .

# Run as a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "app.py"]
EXPOSE 8080

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
