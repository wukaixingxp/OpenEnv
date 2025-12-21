# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Self-contained Dockerfile for coding_env
# Works with the flattened temp directory structure created by `openenv build`:
#   temp/coding_env/
#   ├── openenv/          # The openenv package (copied from src/openenv)
#   ├── pyproject.toml    # Updated to reference local openenv
#   ├── server/
#   └── ...

FROM python:3.11-slim

# Build arguments
ARG BUILD_MODE=in-repo
ARG ENV_NAME=coding

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy entire build context (the prepared coding_env directory)
COPY . /app/env

WORKDIR /app/env

# Install dependencies using uv
# The pyproject.toml has been updated to reference openenv @ file:///app/env/openenv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable || \
    uv pip install --system -e .

# Install the project itself (openenv-core from repo root)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable || \
    uv pip install --system -e .

# Install the coding_env package from envs/coding_env/
# This is needed when building from repo root (not via openenv build)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -d "/app/env/envs/coding_env" ]; then \
        cd /app/env/envs/coding_env && \
        uv pip install -e . ; \
    fi

# Set PATH to include uv's venv if created
ENV PATH="/app/env/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
# When building from repo root, coding_env is under /app/env/envs/coding_env
# When building via openenv build (flattened), it's directly under /app/env
ENV PYTHONPATH="/app/env:/app/env/envs/coding_env"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
# Use the installed package path (coding_env.server.app) instead of the raw directory path (server.app)
# This ensures relative imports like "from ..models" work correctly
CMD ["python", "-m", "uvicorn", "coding_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
