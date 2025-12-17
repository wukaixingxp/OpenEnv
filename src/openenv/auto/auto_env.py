# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoEnv - Automatic Environment Selection
==========================================

AutoEnv provides a HuggingFace-style API for automatically selecting and
instantiating the correct environment client from installed packages or
HuggingFace Hub.

This module simplifies environment creation by automatically detecting the
environment type from the name and instantiating the appropriate client class.

Example:
    >>> from openenv import AutoEnv, AutoAction
    >>>
    >>> # From installed package
    >>> env = AutoEnv.from_env("coding-env")
    >>>
    >>> # From HuggingFace Hub
    >>> env = AutoEnv.from_env("meta-pytorch/coding-env")
    >>>
    >>> # With configuration
    >>> env = AutoEnv.from_env("coding", env_vars={"DEBUG": "1"})
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import tempfile
import requests
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING, Dict

from ._discovery import get_discovery, _is_hub_url, _normalize_env_name

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider
    from openenv.core.http_env_client import HTTPEnvClient

logger = logging.getLogger(__name__)

# Cache for repo ID → env_name mapping to avoid redundant downloads
_hub_env_name_cache: Dict[str, str] = {}


class AutoEnv:
    """
    AutoEnv automatically selects and instantiates the correct environment client
    based on environment names or HuggingFace Hub repositories.

    This class follows the HuggingFace AutoModel pattern, making it easy to work
    with different environments without needing to import specific client classes.

    The class provides factory methods that:
    1. Check if name is a HuggingFace Hub URL/repo ID
    2. If Hub: download and install the environment package
    3. If local: look up the installed openenv-* package
    4. Import and instantiate the client class

    Example:
        >>> # From installed package
        >>> env = AutoEnv.from_env("coding-env")
        >>>
        >>> # From HuggingFace Hub
        >>> env = AutoEnv.from_env("meta-pytorch/coding-env")
        >>>
        >>> # List available environments
        >>> AutoEnv.list_environments()

    Note:
        AutoEnv is not meant to be instantiated directly. Use the class method
        from_env() instead.
    """

    def __init__(self):
        """AutoEnv should not be instantiated directly. Use class methods instead."""
        raise TypeError(
            "AutoEnv is a factory class and should not be instantiated directly. "
            "Use AutoEnv.from_env() instead."
        )

    @classmethod
    def _resolve_space_url(cls, repo_id: str) -> str:
        """
        Resolve HuggingFace Space repo ID to Space URL.

        Args:
            repo_id: HuggingFace repo ID (e.g., "wukaixingxp/coding-env-test")

        Returns:
            Space URL (e.g., "https://wukaixingxp-coding-env-test.hf.space")

        Examples:
            >>> AutoEnv._resolve_space_url("wukaixingxp/coding-env-test")
            'https://wukaixingxp-coding-env-test.hf.space'
        """
        # Clean up repo_id if it's a full URL
        if "huggingface.co" in repo_id:
            # Extract org/repo from URL
            # https://huggingface.co/wukaixingxp/coding-env-test -> wukaixingxp/coding-env-test
            parts = repo_id.split("/")
            if len(parts) >= 2:
                repo_id = f"{parts[-2]}/{parts[-1]}"

        # Convert user/space-name to user-space-name.hf.space
        space_slug = repo_id.replace("/", "-")
        return f"https://{space_slug}.hf.space"

    @classmethod
    def _check_space_availability(cls, space_url: str, timeout: float = 5.0) -> bool:
        """
        Check if HuggingFace Space is running and accessible.

        Args:
            space_url: Space URL to check
            timeout: Request timeout in seconds

        Returns:
            True if Space is accessible, False otherwise

        Examples:
            >>> AutoEnv._check_space_availability("https://wukaixingxp-coding-env-test.hf.space")
            True
        """
        try:
            # Try to access the health endpoint
            response = requests.get(f"{space_url}/health", timeout=timeout)
            if response.status_code == 200:
                return True
            
            # If health endpoint doesn't exist, try root endpoint
            response = requests.get(space_url, timeout=timeout)
            return response.status_code == 200
        except (requests.RequestException, Exception) as e:
            logger.debug(f"Space {space_url} not accessible: {e}")
            return False

    @classmethod
    def _check_server_availability(cls, base_url: str, timeout: float = 2.0) -> bool:
        """
        Check if a server is running and accessible at the given URL.

        Args:
            base_url: URL to check (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds

        Returns:
            True if server is accessible, False otherwise

        Examples:
            >>> AutoEnv._check_server_availability("http://localhost:8000")
            True
        """
        try:
            # Try to access the health endpoint
            response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
            if response.status_code == 200:
                return True

            # If health endpoint doesn't exist, try root endpoint
            response = requests.get(base_url, timeout=timeout)
            return response.status_code == 200
        except (requests.RequestException, Exception) as e:
            logger.debug(f"Server at {base_url} not accessible: {e}")
            return False

    @classmethod
    def _wait_for_server_ready(
        cls,
        base_url: str,
        timeout: float = 30.0,
        check_interval: float = 0.5,
    ) -> None:
        """
        Wait for server to become ready at the given URL.

        Args:
            base_url: URL to check (e.g., "http://localhost:8000")
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Raises:
            TimeoutError: If server doesn't become ready within timeout
        """
        import time

        start_time = time.time()
        logger.info(f"⏳ Waiting for server at {base_url} to be ready...")

        while time.time() - start_time < timeout:
            if cls._check_server_availability(base_url, timeout=2.0):
                elapsed = time.time() - start_time
                logger.info(f"✅ Server ready at {base_url} (took {elapsed:.1f}s)")
                return

            time.sleep(check_interval)

        raise TimeoutError(
            f"Server at {base_url} did not become ready within {timeout}s"
        )

    @classmethod
    def _start_docker_container(
        cls,
        docker_image: str,
        target_url: str,
        container_provider: Optional[ContainerProvider] = None,
        env_vars: Optional[Dict[str, str]] = None,
        wait_timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Start a Docker container for the environment.

        Args:
            docker_image: Docker image to run
            target_url: Target URL (e.g., "http://localhost:8000")
            container_provider: Optional provider (creates LocalDockerProvider if None)
            env_vars: Environment variables for container
            wait_timeout: Timeout for container startup

        Returns:
            Dict with 'provider', 'base_url' keys

        Raises:
            RuntimeError: If Docker container fails to start
        """
        from urllib.parse import urlparse

        # Parse target URL to get port
        parsed = urlparse(target_url)
        port = parsed.port or 8000

        # Create or use provider
        if container_provider is None:
            # Lazy import to avoid circular dependencies
            from openenv.core.containers.runtime.providers import LocalDockerProvider
            container_provider = LocalDockerProvider()

        # Start container
        logger.info(f"🐳 Starting Docker container: {docker_image}")
        logger.info(f"   Port: {port}")

        try:
            # Start container with specific port
            # provider.start_container returns the base_url
            base_url = container_provider.start_container(
                image=docker_image,
                port=port,
                env_vars=env_vars or {},
            )

            # Wait for container to be ready
            container_provider.wait_for_ready(base_url, timeout_s=wait_timeout)

            logger.info(f"✅ Container ready at {base_url}")

            return {
                "provider": container_provider,
                "base_url": base_url,
            }
        except Exception as e:
            logger.error(f"Failed to start Docker container: {e}")
            raise RuntimeError(f"Failed to start Docker container: {e}") from e

    @classmethod
    def _download_from_hub(
        cls, repo_id: str, cache_dir: Optional[Path] = None
    ) -> Path:
        """
        Download environment from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "meta-pytorch/coding-env")
            cache_dir: Optional cache directory

        Returns:
            Path to downloaded environment directory

        Raises:
            ImportError: If huggingface_hub is not installed
            ValueError: If download fails
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "HuggingFace Hub support requires huggingface_hub package.\n"
                "Install it with: pip install huggingface_hub"
            )

        # Clean up repo_id if it's a full URL
        if "huggingface.co" in repo_id:
            # Extract org/repo from URL
            # https://huggingface.co/meta-pytorch/coding-env -> meta-pytorch/coding-env
            parts = repo_id.split("/")
            if len(parts) >= 2:
                repo_id = f"{parts[-2]}/{parts[-1]}"

        logger.info(f"Downloading environment from HuggingFace Hub: {repo_id}")

        try:
            # Download to cache
            env_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir or Path(tempfile.gettempdir()) / "openenv_hub_cache",
                repo_type="space",  # OpenEnv environments are published as Spaces
            )
            return Path(env_path)
        except Exception as e:
            raise ValueError(
                f"Failed to download environment from HuggingFace Hub: {repo_id}\n"
                f"Error: {e}\n"
                f"Make sure the repository exists and is accessible."
            ) from e

    @classmethod
    def _install_from_path(cls, env_path: Path) -> str:
        """
        Install environment package from a local path.

        Args:
            env_path: Path to environment directory containing pyproject.toml

        Returns:
            Package name that was installed

        Raises:
            ValueError: If installation fails
        """
        if not (env_path / "pyproject.toml").exists():
            raise ValueError(
                f"Environment directory does not contain pyproject.toml: {env_path}"
            )

        logger.info(f"Installing environment from: {env_path}")

        try:
            # Install in editable mode
            subprocess.run(
                ["pip", "install", "-e", str(env_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            # Read package name from pyproject.toml
            import toml

            with open(env_path / "pyproject.toml", "r") as f:
                pyproject = toml.load(f)

            package_name = pyproject.get("project", {}).get("name")
            if not package_name:
                raise ValueError("Could not determine package name from pyproject.toml")

            logger.info(f"Successfully installed: {package_name}")
            return package_name

        except subprocess.CalledProcessError as e:
            raise ValueError(
                f"Failed to install environment package from {env_path}\n"
                f"Error: {e.stderr}"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to install environment package: {e}") from e

    @classmethod
    def _get_package_name_from_hub(cls, name: str) -> tuple[str, Path]:
        """
        Download Space and get the package name from pyproject.toml.
        
        Args:
            name: HuggingFace repo ID (e.g., "wukaixingxp/coding-env-test")
            
        Returns:
            Tuple of (package_name, env_path)
            Example: ("openenv-coding_env", Path("/tmp/..."))
        """
        # Download from Hub
        env_path = cls._download_from_hub(name)
        
        # Read package name from pyproject.toml
        import toml
        
        pyproject_path = env_path / "pyproject.toml"
        if not pyproject_path.exists():
            raise ValueError(
                f"Environment directory does not contain pyproject.toml: {env_path}"
            )
        
        with open(pyproject_path, "r") as f:
            pyproject = toml.load(f)
        
        package_name = pyproject.get("project", {}).get("name")
        if not package_name:
            raise ValueError(
                f"Could not determine package name from pyproject.toml at {pyproject_path}"
            )
        
        return package_name, env_path

    @classmethod
    def _is_package_installed(cls, package_name: str) -> bool:
        """
        Check if a package is already installed.
        
        Args:
            package_name: Package name (e.g., "openenv-coding_env")
            
        Returns:
            True if installed, False otherwise
        """
        try:
            import importlib.metadata
            importlib.metadata.distribution(package_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    @classmethod
    def _ensure_package_from_hub(cls, name: str) -> str:
        """
        Ensure package from HuggingFace Hub is installed.
        
        Only downloads and installs if not already installed.
        Uses a cache to avoid redundant downloads for the same repo ID.
        
        Args:
            name: HuggingFace repo ID (e.g., "wukaixingxp/coding-env-test")
            
        Returns:
            Environment name (e.g., "coding_env")
        """
        global _hub_env_name_cache
        
        # Check if we already resolved this repo ID
        if name in _hub_env_name_cache:
            env_name = _hub_env_name_cache[name]
            logger.debug(f"✅ Using cached env name for {name}: {env_name}")
            return env_name
        
        # Download and get actual package name from pyproject.toml
        logger.info(f"📦 Checking package from HuggingFace Space...")
        package_name, env_path = cls._get_package_name_from_hub(name)
        
        # Check if already installed
        if cls._is_package_installed(package_name):
            logger.info(f"✅ Package already installed: {package_name}")
            # Clear and refresh discovery cache to make sure it's detected
            get_discovery().clear_cache()
            get_discovery().discover(use_cache=False)
        else:
            # Not installed, install it now
            logger.info(f"📦 Package not found, installing: {package_name}")
            cls._install_from_path(env_path)
            # Clear discovery cache to pick up the newly installed package
            get_discovery().clear_cache()
        
        # Extract environment name from package name
        # "openenv-coding_env" -> "coding_env"
        env_name = package_name.replace("openenv-", "").replace("-", "_")
        
        # Cache the result to avoid redundant downloads
        _hub_env_name_cache[name] = env_name
        
        return env_name

    @classmethod
    def from_env(
        cls,
        name: str,
        base_url: Optional[str] = "http://localhost:8000",
        docker_image: Optional[str] = None,
        container_provider: Optional[ContainerProvider] = None,
        wait_timeout: float = 30.0,
        env_vars: Optional[Dict[str, str]] = None,
        auto_start_docker: bool = True,
        **kwargs: Any,
    ) -> HTTPEnvClient:
        """
        Create an environment client from a name or HuggingFace Hub repository.

        This method automatically:
        1. Checks if the name is a HuggingFace Hub URL/repo ID
        2. If Hub: downloads and installs the environment package
        3. If local: looks up the installed openenv-* package
        4. Imports the client class and instantiates it
        5. Smart fallback: Tries base_url first, then auto-starts Docker if needed

        Args:
            name: Environment name or HuggingFace Hub repo ID
                  Examples:
                  - "coding" / "coding-env" / "coding_env"
                  - "meta-pytorch/coding-env" (Hub repo ID)
                  - "https://huggingface.co/meta-pytorch/coding-env" (Hub URL)
            base_url: Base URL for HTTP connection. Defaults to "http://localhost:8000".
                     Set to None to skip connection attempt and use Docker directly.
                     With auto_start_docker=True: tries connection first, starts Docker if fails.
            docker_image: Optional Docker image name (overrides default).
                         Used for Docker fallback or when base_url=None.
            container_provider: Optional container provider.
                               Used for Docker fallback or when base_url=None.
            wait_timeout: Timeout for container startup (seconds).
                         Used for Docker fallback or when base_url=None.
            env_vars: Optional environment variables for the container.
                     Used for Docker fallback or when base_url=None.
            auto_start_docker: Enable Docker fallback if server not running (default: True).
                              Set to False to fail fast without Docker attempt.
            **kwargs: Additional arguments passed to the client class

        Returns:
            Instance of the environment client class

        Raises:
            ValueError: If environment not found or cannot be loaded
            ImportError: If environment package is not installed

        Examples:
            >>> # Default: tries localhost:8000, falls back to Docker if not running
            >>> env = AutoEnv.from_env("coding")
            >>>
            >>> # Custom base URL with Docker fallback
            >>> env = AutoEnv.from_env("coding", base_url="http://localhost:8001")
            >>>
            >>> # From HuggingFace Hub (auto-detects Space URL)
            >>> env = AutoEnv.from_env("wukaixingxp/coding-env-test")
            >>>
            >>> # Disable Docker fallback (fail fast if server not running)
            >>> env = AutoEnv.from_env("coding", auto_start_docker=False)
            >>>
            >>> # Skip connection attempt, use Docker directly
            >>> env = AutoEnv.from_env("coding", base_url=None)
            >>>
            >>> # With environment variables
            >>> env = AutoEnv.from_env(
            ...     "dipg",
            ...     env_vars={"DIPG_DATASET_PATH": "/data/dipg"}
            ... )
        """
        # Check if it's a HuggingFace Hub URL or repo ID
        if _is_hub_url(name):
            # Try to connect to Space directly first
            space_url = cls._resolve_space_url(name)
            logger.info(f"Checking if HuggingFace Space is accessible: {space_url}")

            space_is_available = cls._check_space_availability(space_url)

            # Only use Space URL if user didn't explicitly provide a different base_url
            if space_is_available and base_url == "http://localhost:8000":
                # Space is accessible! We'll connect directly without Docker
                logger.info(f"✅ Space is accessible at: {space_url}")
                logger.info("📦 Installing package for client code (no Docker needed)...")

                # Ensure package is installed (downloads only if needed)
                env_name = cls._ensure_package_from_hub(name)

                # Set base_url to connect to remote Space
                base_url = space_url
                logger.info(f"🚀 Will connect to remote Space (no local Docker)")
            else:
                # Space not accessible or user provided explicit base_url
                if not space_is_available:
                    logger.info(f"❌ Space not accessible at {space_url}")
                    if base_url == "http://localhost:8000":
                        logger.info("📦 Falling back to localhost:8000...")
                    else:
                        logger.info(f"📦 Will use provided base_url: {base_url}")

                # Ensure package is installed (downloads only if needed)
                env_name = cls._ensure_package_from_hub(name)
        else:
            env_name = name

        # Get environment info from discovery
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(env_name)

        if not env_info:
            # Environment not found - provide helpful error message
            available_envs = discovery.discover()

            if not available_envs:
                raise ValueError(
                    f"No OpenEnv environments found.\n"
                    f"Install an environment with: pip install openenv-<env-name>\n"
                    f"Or specify a HuggingFace Hub repository: AutoEnv.from_env('org/repo')"
                )

            # Try to suggest similar environment names
            from difflib import get_close_matches

            env_keys = list(available_envs.keys())
            suggestions = get_close_matches(env_name, env_keys, n=3, cutoff=0.6)

            error_msg = f"Unknown environment '{env_name}'.\n"
            if suggestions:
                error_msg += f"Did you mean: {', '.join(suggestions)}?\n"
            error_msg += f"Available environments: {', '.join(sorted(env_keys))}"

            raise ValueError(error_msg)

        # Get the client class
        try:
            client_class = env_info.get_client_class()
        except ImportError as e:
            raise ImportError(
                f"Failed to import environment client for '{env_name}'.\n"
                f"Package '{env_info.package_name}' appears to be installed but the module cannot be imported.\n"
                f"Try reinstalling: pip install --force-reinstall {env_info.package_name}\n"
                f"Original error: {e}"
            ) from e

        # Determine Docker image to use
        if docker_image is None:
            docker_image = env_info.default_image

        # Create client instance with smart fallback
        try:
            if base_url:
                # Smart fallback: try connection first, then Docker if needed

                # Step 1: Check if server is already running
                server_running = cls._check_server_availability(base_url, timeout=2.0)

                if server_running:
                    # Server is running - connect directly
                    logger.info(f"✅ Connecting to running server at {base_url}")
                    return client_class(base_url=base_url, provider=None, **kwargs)

                # Step 2: Server not running
                logger.info(f"⚠️  Server not running at {base_url}")

                if not auto_start_docker:
                    # User disabled Docker fallback - fail fast
                    raise ConnectionError(
                        f"Server not running at {base_url} and auto_start_docker=False.\n"
                        f"\n"
                        f"Solutions:\n"
                        f"1. Start the server manually:\n"
                        f"   cd /path/to/OpenEnv\n"
                        f"   python -m envs.{env_name}.server.app\n"
                        f"\n"
                        f"2. Enable Docker fallback: auto_start_docker=True (default)\n"
                        f"3. Use explicit Docker mode: base_url=None"
                    )

                # Step 3: Attempt Docker auto-start fallback
                logger.info(f"🐳 Attempting Docker auto-start: {docker_image}")

                try:
                    # Start Docker container
                    container_info = cls._start_docker_container(
                        docker_image=docker_image,
                        target_url=base_url,
                        container_provider=container_provider,
                        env_vars=env_vars,
                        wait_timeout=wait_timeout,
                    )

                    # Connect to now-running container
                    # Use the actual base_url from the container (might differ from requested)
                    actual_base_url = container_info["base_url"]
                    logger.info(f"✅ Docker container started and ready at {actual_base_url}")
                    return client_class(
                        base_url=actual_base_url,
                        provider=container_info["provider"],
                        **kwargs
                    )

                except Exception as docker_error:
                    # Docker auto-start failed
                    raise ValueError(
                        f"Failed to start environment for '{env_name}'.\n"
                        f"Server not running at {base_url} and Docker auto-start failed.\n"
                        f"Docker error: {docker_error}\n"
                        f"\n"
                        f"Solutions:\n"
                        f"1. Start the server manually:\n"
                        f"   cd /path/to/OpenEnv\n"
                        f"   export PYTHONPATH=\"${{PWD}}/src:${{PYTHONPATH}}\"\n"
                        f"   python -m envs.{env_name}.server.app\n"
                        f"\n"
                        f"2. Fix Docker/Podman configuration\n"
                        f"3. Check if port {base_url.split(':')[-1] if ':' in base_url else '8000'} is available"
                    ) from docker_error

            else:
                # Explicit None - use Docker mode directly (no connection attempt)
                logger.info(f"🐳 Starting Docker container (base_url=None)")
                return client_class.from_docker_image(
                    image=docker_image,
                    provider=container_provider,
                    wait_timeout=wait_timeout,
                    env_vars=env_vars or {},
                    **kwargs,
                )

        except Exception as e:
            # Final catch-all for unexpected errors
            if isinstance(e, (ValueError, ConnectionError)):
                # Re-raise our own errors
                raise
            raise ValueError(
                f"Failed to create environment client for '{env_name}'.\n"
                f"Client class: {client_class.__name__}\n"
                f"Docker image: {docker_image}\n"
                f"Error: {e}"
            ) from e

    @classmethod
    def from_name(cls, *args, **kwargs) -> HTTPEnvClient:
        """
        Alias for from_env() for backwards compatibility.

        This method is deprecated. Use from_env() instead.

        Examples:
            >>> # Old way (still works)
            >>> env = AutoEnv.from_name("coding-env")
            >>>
            >>> # New way (recommended)
            >>> env = AutoEnv.from_env("coding-env")
        """
        import warnings
        warnings.warn(
            "from_name() is deprecated. Use from_env() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return cls.from_env(*args, **kwargs)

    @classmethod
    def get_env_class(cls, name: str):
        """
        Get the environment client class without instantiating it.

        Args:
            name: Environment name

        Returns:
            The environment client class

        Raises:
            ValueError: If environment not found

        Examples:
            >>> CodingEnv = AutoEnv.get_env_class("coding")
            >>> # Now you can instantiate it yourself
            >>> env = CodingEnv(base_url="http://localhost:8000")
        """
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(name)

        if not env_info:
            raise ValueError(f"Unknown environment: {name}")

        return env_info.get_client_class()

    @classmethod
    def get_env_info(cls, name: str) -> Dict[str, Any]:
        """
        Get detailed information about an environment.

        Args:
            name: Environment name

        Returns:
            Dictionary with environment metadata

        Raises:
            ValueError: If environment not found

        Examples:
            >>> info = AutoEnv.get_env_info("coding")
            >>> print(info['description'])
            'Coding environment for OpenEnv'
            >>> print(info['default_image'])
            'coding-env:latest'
        """
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(name)

        if not env_info:
            raise ValueError(f"Unknown environment: {name}")

        return {
            "env_key": env_info.env_key,
            "name": env_info.name,
            "package": env_info.package_name,
            "version": env_info.version,
            "description": env_info.description,
            "env_class": env_info.client_class_name,
            "action_class": env_info.action_class_name,
            "observation_class": env_info.observation_class_name,
            "module": env_info.client_module_path,
            "default_image": env_info.default_image,
            "spec_version": env_info.spec_version,
        }

    @classmethod
    def list_environments(cls) -> None:
        """
        Print a formatted list of all available environments.

        This discovers all installed openenv-* packages and displays
        their metadata in a user-friendly format.

        Examples:
            >>> AutoEnv.list_environments()
            Available OpenEnv Environments:
            ----------------------------------------------------------------------
              echo           : Echo Environment (v0.1.0)
                               Package: openenv-echo-env
              coding         : Coding Environment (v0.1.0)
                               Package: openenv-coding_env
            ----------------------------------------------------------------------
            Total: 2 environments
        """
        discovery = get_discovery()
        discovery.list_environments()
