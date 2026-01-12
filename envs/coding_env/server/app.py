# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Coding Environment.

This module creates an HTTP server that exposes the PythonCodeActEnv
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.coding_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.coding_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.coding_env.server.app

    # With custom imports:
    PYTHON_ADDITIONAL_IMPORTS="numpy,pandas,matplotlib" python -m envs.coding_env.server.app
"""

import os

from openenv.core.env_server import create_app

from coding_env.models import CodeAction, CodeObservation
from coding_env.server.python_codeact_env import PythonCodeActEnv

# Get additional imports from environment variable
# Format: comma-separated list, e.g., "numpy,pandas,matplotlib"
additional_imports_str = os.environ.get("PYTHON_ADDITIONAL_IMPORTS", "")
if additional_imports_str:
    additional_imports = [imp.strip() for imp in additional_imports_str.split(",") if imp.strip()]
else:
    # Default imports that match the common_imports used in reward evaluation
    additional_imports = [
        "sys",
        "os",
        "functools",
        "typing",
    ]


# Create a factory function that creates PythonCodeActEnv with additional_imports
def create_env():
    return PythonCodeActEnv(additional_imports=additional_imports)


# Create the app with web interface and README integration
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(create_env, CodeAction, CodeObservation, env_name="coding_env")


def main():
    """Main entry point for running the server."""
    import sys
    import uvicorn

    # Get port from environment variable or command line argument
    # Priority: command line arg > environment variable > default (8000)
    port = int(os.environ.get("PORT", 8000))

    # Override with command line argument if provided
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port argument: {sys.argv[1]}, using port {port}")

    print(f"Starting server on port {port}")
    print(f"Additional imports: {additional_imports}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
