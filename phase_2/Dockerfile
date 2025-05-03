# Base image with Python
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime


# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
