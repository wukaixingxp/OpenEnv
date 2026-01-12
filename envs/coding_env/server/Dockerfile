# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/env

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Set default port (can be overridden via environment variable)
ENV PORT=8000

# Set default additional imports (can be overridden via environment variable)
ENV PYTHON_ADDITIONAL_IMPORTS=""

# Expose port (note: actual port is determined by PORT env var)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Health check (uses PORT env variable)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the server (uses PORT env variable)
CMD sh -c "uvicorn coding_env.server.app:app --host 0.0.0.0 --port ${PORT}"
