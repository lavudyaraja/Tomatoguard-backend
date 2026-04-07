# Use a high-speed, lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# We use torch-cpu to save massive amounts of RAM
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces (and most other cloud providers) listen on port 7860 by default
# But we can override this in our main.py or CMD
EXPOSE 8000

# Start the FastAPI engine
# We use --host 0.0.0.0 to ensure the container is reachable
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
