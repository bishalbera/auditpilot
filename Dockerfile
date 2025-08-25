# Multi-stage Dockerfile for Compliance Copilot
# Stage 1: Base Python image with system dependencies
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create app user with proper home directory
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Set work directory
WORKDIR /app

# Set up npm for non-root user
RUN mkdir -p /home/appuser/.npm-global && \
    chown -R appuser:appuser /home/appuser/.npm-global
ENV NPM_CONFIG_PREFIX=/home/appuser/.npm-global
ENV PATH=/home/appuser/.npm-global/bin:$PATH

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY . .

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p templates static config logs data

# Set ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["run.py"]
