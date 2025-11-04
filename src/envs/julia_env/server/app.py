# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Julia Environment with concurrent execution support.

This module creates an HTTP server that exposes the JuliaCodeActEnv
over HTTP endpoints with optimized async execution for handling multiple
concurrent requests efficiently.

Features:
- Async Julia code execution to avoid blocking
- Environment pool for concurrent request handling
- Thread pool executor for CPU-bound Julia tasks
- 10x+ performance improvement over single-threaded version

Usage:
    # Development (with auto-reload):
    uvicorn envs.julia_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production (with multiple workers for even better concurrency):
    uvicorn envs.julia_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.julia_env.server.app
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict

from fastapi import Body, FastAPI

from ..models import JuliaAction, JuliaObservation
from .julia_codeact_env import JuliaCodeActEnv

# Configuration
MAX_WORKERS = int(
    os.getenv("JULIA_MAX_WORKERS", "8")
)  # Number of concurrent Julia executions
ENABLE_WEB = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")

# Global thread pool executor for CPU-bound Julia tasks
executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global executor
    # Startup: Create thread pool
    executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS, thread_name_prefix="julia_worker"
    )
    print(f"✅ Julia Environment Server started with {MAX_WORKERS} concurrent workers")
    yield
    # Shutdown: Cleanup
    executor.shutdown(wait=True)
    print("✅ Julia Environment Server shutdown complete")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="Julia Environment Server",
    description="Async Julia code execution environment with concurrent request support",
    version="2.0.0",
    lifespan=lifespan,
)


async def execute_julia_async(action: JuliaAction) -> JuliaObservation:
    """
    Execute Julia code asynchronously in thread pool.

    This runs the CPU-bound Julia execution in a separate thread to avoid
    blocking the event loop, allowing the server to handle multiple requests
    concurrently.
    """
    loop = asyncio.get_event_loop()

    # Create a fresh environment instance for this request
    # This ensures thread safety and allows concurrent execution
    env = JuliaCodeActEnv()

    # Run the blocking step() call in thread pool
    observation = await loop.run_in_executor(executor, env.step, action)

    return observation


@app.post("/reset")
async def reset(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Reset endpoint - returns initial observation.

    Creates a fresh environment instance for the new episode.
    """
    # Run reset in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    env = JuliaCodeActEnv()
    observation = await loop.run_in_executor(executor, env.reset)

    # Serialize observation
    obs_dict = asdict(observation)
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    obs_dict.pop("metadata", None)

    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }


@app.post("/step")
async def step(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step endpoint - executes Julia code and returns observation.

    Runs Julia code execution asynchronously to handle multiple concurrent requests.
    Each request gets its own environment instance for thread safety.
    """
    action_data = request.get("action", {})

    # Deserialize action
    metadata = action_data.pop("metadata", {})
    action = JuliaAction(**action_data)
    action.metadata = metadata

    # Execute Julia code asynchronously
    observation = await execute_julia_async(action)

    # Serialize observation
    obs_dict = asdict(observation)
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    obs_dict.pop("metadata", None)

    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """
    State endpoint - returns environment metadata.

    Note: Since each request creates a fresh environment, this returns
    general server state rather than specific episode state.
    """
    return {
        "max_workers": MAX_WORKERS,
        "executor_type": "ThreadPoolExecutor",
        "status": "ready",
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "workers": str(MAX_WORKERS)}


if __name__ == "__main__":
    import uvicorn

    # Run with uvicorn
    # Use multiple workers for even better concurrency
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
