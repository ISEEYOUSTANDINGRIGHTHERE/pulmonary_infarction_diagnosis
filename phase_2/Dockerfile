# Use slim Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system packages needed for DICOM, image handling, visualization, etc.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache dependencies better
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
