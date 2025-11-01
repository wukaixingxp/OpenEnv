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
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
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
            port: Port to expose (if None, uses 8000)
            env_vars: Environment variables for the container
            **kwargs: Additional Docker run options
                - command_override: List of command args to override container CMD

        Returns:
            Base URL to connect to the container
        """
        import subprocess
        import time

        # Use default port if not specified
        if port is None:
            port = 8000

        # Generate container name
        self._container_name = self._generate_container_name(image)

        # Build docker run command
        # Use host networking for better performance and consistency with podman
        cmd = [
            "docker", "run",
            "-d",  # Detached
            "--name", self._container_name,
            "--network", "host",  # Use host network
        ]

        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Add image
        cmd.append(image)
          
        # Add command override if provided (to change port)
        if "command_override" in kwargs:
            cmd.extend(kwargs["command_override"])
        elif port != 8000:
            # Infer app path from image name for common environments
            app_module = self._infer_app_module(image)
            if app_module:
                cmd.extend([
                    "uvicorn",
                    app_module,
                    "--host", "0.0.0.0",
                    "--port", str(port)
                ])

        # Run container
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self._container_id = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to start Docker container.\nCommand: {' '.join(cmd)}\nExit code: {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}"
            raise RuntimeError(error_msg) from e

        # Wait a moment for container to start
        time.sleep(1)

        base_url = f"http://127.0.0.1:{port}"
        return base_url

    def stop_container(self) -> None:
        """
        Stop and remove the Docker container.
        """
        if self._container_id is None:
            return

        import subprocess

        try:
            # Stop container
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
                check=True,
                timeout=10,
            )

            # Remove container
            subprocess.run(
                ["docker", "rm", self._container_id],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except subprocess.CalledProcessError:
            # Container might already be stopped/removed
            pass
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
        import subprocess
        import logging

        start_time = time.time()
        health_url = f"{base_url}/health"
        last_error = None

        while time.time() - start_time < timeout_s:
            try:
                response = requests.get(health_url, timeout=2.0)
                if response.status_code == 200:
                    return
            except requests.RequestException as e:
                last_error = str(e)

            time.sleep(0.5)

        # If we timeout, provide diagnostic information
        error_msg = f"Container at {base_url} did not become ready within {timeout_s}s"
        
        if self._container_id:
            try:
                # Get container logs to help diagnose the issue
                result = subprocess.run(
                    ["docker", "logs", "--tail", "50", self._container_id],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout or result.stderr:
                    error_msg += f"\n\nContainer logs (last 50 lines):\n{result.stdout}\n{result.stderr}"
            except Exception:
                pass

        if last_error:
            error_msg += f"\n\nLast connection error: {last_error}"

        raise TimeoutError(error_msg)

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

    def _infer_app_module(self, image: str) -> Optional[str]:
        """
        Infer the uvicorn app module path from the image name.

        Args:
            image: Container image name

        Returns:
            App module path like "envs.coding_env.server.app:app" or None
        """
        clean_image = image.split("/")[-1].split(":")[0]
        
        # Map common environment names to their app modules
        env_module_map = {
            "coding-env": "envs.coding_env.server.app:app",
            "echo-env": "envs.echo_env.server.app:app",
            "git-env": "envs.git_env.server.app:app",
            "openspiel-env": "envs.openspiel_env.server.app:app",
            "sumo-rl-env": "envs.sumo_rl_env.server.app:app",
            "finrl-env": "envs.finrl_env.server.app:app",
        }
        
        return env_module_map.get(clean_image)



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
