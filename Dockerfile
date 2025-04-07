FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/test_model_cache/gemma-3-1b-it-q4_0.gguf \
    MCP_CONFIG_PATH=/app/mcp.json \
    LOG_LEVEL=INFO \
    NODE_VERSION=18.18.0

# Set working directory
WORKDIR /app

# Install build dependencies, Python packages, and Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    curl \
    ca-certificates \
    gnupg \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    # Install Node.js based on architecture
    && ARCH=$(dpkg --print-architecture) \
    && case "${ARCH}" in \
       aarch64|arm64) \
         NODEARCH='arm64' \
         ;; \
       amd64|x86_64) \
         NODEARCH='x64' \
         ;; \
       *) \
         echo "Unsupported architecture: ${ARCH}" \
         exit 1 \
         ;; \
    esac \
    && curl -fsSLO --compressed "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODEARCH}.tar.gz" \
    && tar -xzf "node-v${NODE_VERSION}-linux-${NODEARCH}.tar.gz" -C /usr/local --strip-components=1 \
    && rm "node-v${NODE_VERSION}-linux-${NODEARCH}.tar.gz" \
    && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
    # Verify Node.js installation
    && node --version \
    && npm --version

# Copy requirements and install dependencies
COPY requirements.txt .
# Install llama-cpp-python with CPU support (no CUDA by default in container)
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && \
    apt-get purge -y gcc g++ python3-dev cmake && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create directories
RUN mkdir -p test_model_cache logs

# Add model download script
COPY download_model.py .
RUN echo "#!/bin/sh\n\
if [ ! -f \$MODEL_PATH ]; then\n\
  echo \"Model file not found. Starting download...\"\n\
  python download_model.py\n\
fi\n\
\n\
# Start application\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8000\n\
" > /app/startup.sh && chmod +x /app/startup.sh

# Expose port
EXPOSE 8000

# Command to run the application with model check on startup
CMD ["/app/startup.sh"] 