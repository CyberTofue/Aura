FROM python:3.10-slim

WORKDIR /app

# 1. Install system build tools (Crucial for -slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy model files and server code
COPY server.py .
COPY image_checkpoint.pth .
COPY chat_checkpoint.pth .
COPY tokenizer.model .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]