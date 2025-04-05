# Use a slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# System dependencies for OpenCV, SimpleITK, and image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH="/app"

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.api.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
