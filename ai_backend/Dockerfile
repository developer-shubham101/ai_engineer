FROM python:3.11-slim

WORKDIR /app

# System deps for audio/vision libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libsndfile1 ffmpeg tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create runtime directories
RUN mkdir -p database logs chroma_storage user_uploaded_files/charts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
