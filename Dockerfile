# Use a high-speed, lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces use UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Hugging Face Spaces listen on port 7860 by default
EXPOSE 7860

# Start the FastAPI engine
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
