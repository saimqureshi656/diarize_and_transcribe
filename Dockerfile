FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip3 install -r requirements.txt

# Copy application code
# Create directories for uploads and logs
RUN mkdir -p /app/uploads /app/logs /app/outputs

# Expose ports
EXPOSE 8001 8501

# Default command (will be overridden by docker-compose)
CMD ["python3", "main.py"]
