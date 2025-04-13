FROM node:18-alpine AS builder

WORKDIR /app/frontend

# Copy package.json and package-lock.json first for caching
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY frontend/ .

# Build the frontend application
RUN npm run build

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for models and static files
RUN mkdir -p /app/models /app/static

COPY --from=builder /app/frontend/dist /app/static

COPY app/ /app/app/
COPY run.sh .
COPY mcp.json .
COPY react_output.gbnf .

ARG MODEL_FILENAME_ARG

COPY models/${MODEL_FILENAME_ARG} /app/models/

ENV MODEL_DIR=/app/models
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]