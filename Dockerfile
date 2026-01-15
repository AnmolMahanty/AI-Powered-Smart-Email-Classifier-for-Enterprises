# Stage 1: Build React Frontend
FROM node:18-slim AS builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
# RUN apt-get update && apt-get install -y ...

# Copy Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Models and Code (Exclude frontend folder to keep context small if possible, but simpler to copy root)
# We copy specific files to avoid clutter
COPY app/ ./app/
COPY final_urgency_model/ ./final_urgency_model/
COPY final_bert_model/ ./final_bert_model/
COPY hybrid_inference.py .
COPY rule_based_urgency.py .

# Copy Built Frontend from Stage 1 to FastAPI Static Dir
COPY --from=builder /app/dist ./app/static

# Expose Port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Run Command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
