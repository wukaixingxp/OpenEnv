# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy openenv-core source (build context should be repo root)
COPY src ./openenv_src/src
COPY pyproject.toml ./openenv_src/
# Create a minimal README.md to satisfy setup
RUN echo "# OpenEnv Core" > ./openenv_src/README.md

# Copy coding_env source
COPY envs/coding_env ./coding_env/

# Install openenv-core from local source
RUN pip install --no-cache-dir -e ./openenv_src/

# Install the coding_env package
RUN pip install --no-cache-dir -e ./coding_env/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Run the server
CMD ["python", "-m", "uvicorn", "coding_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
