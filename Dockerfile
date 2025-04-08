# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV RUNNING_IN_DOCKER True
# Set default log level for the container
ENV LOG_LEVEL INFO
# Environment variable for Hugging Face token (optional, can be overridden at runtime)
ENV HUGGING_FACE_TOKEN ""

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including Node.js (for npx) and build essentials
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg build-essential git \
    # Install Node.js LTS
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    # Clean up APT when done
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create directories for models and logs if they don't exist (permissions might need adjustment)
# Ensure the user running the container has write permissions or run as root/adjust permissions later
RUN mkdir -p /app/models /app/logs && chmod -R 777 /app/models /app/logs

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["python", "-m", "app.main", "--host", "0.0.0.0", "--port", "8000"] 