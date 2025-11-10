FROM python:3.10-slim

# OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + model files
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "app.py"]
