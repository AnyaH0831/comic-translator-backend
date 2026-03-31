# Use Python 3.10 as the base (best for PaddleOCR)
FROM python:3.10-slim

# Install Linux system dependencies for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
# Copy the rest of your code
COPY . .

# Expose the port Render uses
EXPOSE 8000

# Start command using the $PORT environment variable
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
