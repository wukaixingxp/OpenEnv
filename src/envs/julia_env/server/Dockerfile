# Copyright (c) Yogesh Singla, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the standard openenv base image
# Built from: docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
# In GitHub Actions, this is overridden to use the GHCR base image

# Use the standard openenv base image
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install Julia using juliaup (official installer - more reliable in Docker)
# Also install build-essential for PackageCompiler (needs gcc to build sysimages)
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install juliaup and Julia
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.10

# Add Julia to PATH
ENV PATH="/root/.juliaup/bin:${PATH}"

# Verify Julia installation
RUN julia --version

# Precompile commonly used Julia packages (Test is built-in, but precompile it)
RUN julia -e 'using Test; println("Julia Test module ready")'

# PERFORMANCE BOOST: Build custom sysimage with precompiled Test module
# This speeds up Julia execution by 10-20x (takes ~2 minutes to build)
RUN echo "Building custom Julia sysimage for faster execution..." && \
    julia -e 'using Pkg; Pkg.add("PackageCompiler")' && \
    mkdir -p /root/.julia/sysimages && \
    julia -e ' \
        using PackageCompiler; \
        precompile_script = """ \
            using Test \
            @test 1 + 1 == 2 \
            function example(a, b) \
                return a + b \
            end \
            @test example(2, 3) == 5 \
        """; \
        write("/tmp/precompile.jl", precompile_script); \
        create_sysimage( \
            [:Test], \
            sysimage_path="/root/.julia/sysimages/julia_with_test.so", \
            precompile_execution_file="/tmp/precompile.jl", \
            cpu_target="generic" \
        ); \
        rm("/tmp/precompile.jl"); \
    ' && \
    echo "✅ Custom sysimage built successfully"

# Set environment variable to use custom sysimage by default
ENV JULIA_SYSIMAGE="/root/.julia/sysimages/julia_with_test.so"

# Install smolagents for Python code execution utilities
RUN pip install --no-cache-dir smolagents

# Copy only what's needed for the Julia environment
COPY src/core/ /app/src/core/
COPY src/envs/julia_env/ /app/src/envs/julia_env/

# Environment variables for port and workers with defaults
ENV PORT=8000
ENV NUM_WORKER=4

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the FastAPI server
CMD uvicorn envs.julia_env.server.app:app --host 0.0.0.0 --port ${PORT} --workers ${NUM_WORKER}
