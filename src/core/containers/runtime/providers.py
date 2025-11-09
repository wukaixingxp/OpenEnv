# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Container provider abstractions for running environment servers.

This module provides a pluggable architecture for different container providers
(local Docker, Kubernetes, cloud providers, etc.) to be used with HTTPEnvClient.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ContainerProvider(ABC):
    """
    Abstract base class for container providers.

    Providers implement this interface to support different container platforms:
    - LocalDockerProvider: Runs containers on local Docker daemon
    - KubernetesProvider: Runs containers in Kubernetes cluster
    - FargateProvider: Runs containers on AWS Fargate
    - CloudRunProvider: Runs containers on Google Cloud Run

    The provider manages a single container lifecycle and provides the base URL
    for connecting to it.

    Example:
        >>> provider = LocalDockerProvider()
        >>> base_url = provider.start_container("echo-env:latest")
        >>> print(base_url)  # http://localhost:8000
        >>> # Use the environment via base_url
        >>> provider.stop_container()
    """

    @abstractmethod
    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Start a container from the specified image.

        Args:
            image: Container image name (e.g., "echo-env:latest")
            port: Port to expose (if None, provider chooses)
            env_vars: Environment variables to pass to container
            **kwargs: Provider-specific options

        Returns:
            Base URL to connect to the container (e.g., "http://localhost:8000")

        Raises:
            RuntimeError: If container fails to start
        """
        pass

    @abstractmethod
    def stop_container(self) -> None:
        """
        Stop and remove the running container.

        This cleans up the container that was started by start_container().
        """
        pass

    @abstractmethod
    def wait_for_ready(self, base_url: str, timeout_s: float = 30.0) -> None:
        """
        Wait for the container to be ready to accept requests.

        This typically polls the /health endpoint until it returns 200.

        Args:
            base_url: Base URL of the container
            timeout_s: Maximum time to wait

        Raises:
            TimeoutError: If container doesn't become ready in time
        """
        pass


class LocalDockerProvider(ContainerProvider):
    """
    Container provider for local Docker daemon.

    This provider runs containers on the local machine using Docker.
    Useful for development and testing.

    Example:
        >>> provider = LocalDockerProvider()
        >>> base_url = provider.start_container("echo-env:latest")
        >>> # Container running on http://localhost:<random-port>
        >>> provider.stop_container()
    """

    def __init__(self):
        """Initialize the local Docker provider."""
        self._container_id: Optional[str] = None
        self._container_name: Optional[str] = None

        # Check if Docker is available
        import subprocess

        try:
            subprocess.run(
                ["docker", "version"],
                check=True,
                capture_output=True,
                timeout=5,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            raise RuntimeError(
                "Docker is not available. Please install Docker Desktop or Docker Engine."
            )

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Start a Docker container locally.

        Args:
            image: Docker image name
            port: Port to expose (if None, finds available port)
            env_vars: Environment variables for the container
            **kwargs: Additional Docker run options
                - memory_gb: Memory limit in GB (default: 4GB)
                - command_override: List of command args to override container CMD

        Returns:
            Base URL to connect to the container
        """
        import subprocess
        import time
        import logging

        logger = logging.getLogger(__name__)

        # Find available port if not specified
        if port is None:
            port = self._find_available_port()

        # Use default memory limit if not specified
        memory_gb = kwargs.get("memory_gb", 16)

        # Generate container name
        self._container_name = self._generate_container_name(image)

        # Build docker run command
        # Use host networking for better performance and consistency with podman
        # NOTE: Do NOT use --rm initially - if container fails to start, we need logs
        cmd = [
            "docker",
            "run",
            "-d",  # Detached
            "--name",
            self._container_name,
            "--network",
            "host",  # Use host network
            "--memory",
            f"{memory_gb}g",  # Limit container memory
            "--memory-swap",
            f"{memory_gb}g",  # Prevent swap usage (set equal to --memory)
            "--oom-kill-disable=false",  # Allow OOM killer (exit gracefully)
        ]

        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Pass custom port via environment variable instead of overriding command
        # This allows the container to use its proper entrypoint/CMD
        if port != 8000:
            cmd.extend(["-e", f"PORT={port}"])

        # Add image
        cmd.append(image)

        # Add command override if provided (explicit override by user)
        if "command_override" in kwargs:
            cmd.extend(kwargs["command_override"])

        # Run container
        try:
            logger.debug(f"Starting container with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._container_id = result.stdout.strip()
            logger.debug(f"Container started with ID: {self._container_id}")
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to start Docker container.\nCommand: {' '.join(cmd)}\nExit code: {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}"
            raise RuntimeError(error_msg) from e

        # Wait a moment for container to start
        time.sleep(1)

        base_url = f"http://localhost:{port}"
        return base_url

    def stop_container(self) -> None:
        """
        Stop and remove the Docker container.
        """
        if self._container_id is None:
            return

        import subprocess

        try:
            # Try graceful stop first (with longer timeout)
            print(f"Stopping container {self._container_id[:12]}...")
            try:
                subprocess.run(
                    ["docker", "stop", "-t", "5", self._container_id],
                    capture_output=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                # If graceful stop times out, force kill
                print(f"Graceful stop timed out, forcing kill...")
                subprocess.run(
                    ["docker", "kill", self._container_id],
                    capture_output=True,
                    timeout=10,
                )

            # Remove container
            print(f"Removing container {self._container_id[:12]}...")
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
                timeout=15,
            )

            print(f"✓ Container cleaned up successfully")

        except subprocess.TimeoutExpired:
            # Last resort: force remove
            print(f"Remove timed out, trying force remove...")
            try:
                subprocess.run(
                    ["docker", "rm", "-f", self._container_id],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass
        except Exception as e:
            # Log but don't fail - container might already be gone
            print(f"Note: Cleanup had issues (container may already be removed): {e}")
        finally:
            self._container_id = None
            self._container_name = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 30.0) -> None:
        """
        Wait for container to be ready by polling /health endpoint.

        Args:
            base_url: Base URL of the container
            timeout_s: Maximum time to wait

        Raises:
            TimeoutError: If container doesn't become ready
        """
        import time
        import requests

        start_time = time.time()
        health_url = f"{base_url}/health"

        while time.time() - start_time < timeout_s:
            try:
                response = requests.get(health_url, timeout=2.0)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass

            time.sleep(0.5)

        # Get container logs for debugging
        logs_snippet = ""
        if self._container_id:
            try:
                import subprocess

                result = subprocess.run(
                    ["docker", "logs", "--tail", "20", self._container_id],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout or result.stderr:
                    logs_snippet = "\n\nContainer logs (last 20 lines):\n"
                    logs_snippet += result.stdout + result.stderr
            except Exception:
                pass

        raise TimeoutError(
            f"Container at {base_url} did not become ready within {timeout_s}s. "
            f"The container is still running and will be cleaned up automatically. "
            f"Try increasing wait_timeout (e.g., wait_timeout=60.0 or higher).{logs_snippet}"
        )

    def _find_available_port(self) -> int:
        """
        Find an available port on localhost.

        Returns:
            An available port number
        """
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _generate_container_name(self, image: str) -> str:
        """
        Generate a unique container name based on image name and timestamp.

        Args:
            image: Docker image name

        Returns:
            A unique container name
        """
        import time

        clean_image = image.split("/")[-1].split(":")[0]
        timestamp = int(time.time() * 1000)
        return f"{clean_image}-{timestamp}"


class KubernetesProvider(ContainerProvider):
    """
    Container provider for Kubernetes clusters.

    This provider creates pods in a Kubernetes cluster and exposes them
    via services or port-forwarding.

    Example:
        >>> provider = KubernetesProvider(namespace="envtorch-dev")
        >>> base_url = provider.start_container("echo-env:latest")
        >>> # Pod running in k8s, accessible via service or port-forward
        >>> provider.stop_container()
    """

    pass
