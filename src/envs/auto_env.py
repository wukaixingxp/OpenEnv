# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoEnv - Automatic Environment Selection
==========================================

AutoEnv provides a HuggingFace-style API for automatically selecting and
instantiating the correct environment client based on Docker image names.

This module simplifies environment creation by automatically detecting the
environment type from the Docker image name and instantiating the appropriate
client class.

Example:
    >>> from envs import AutoEnv, AutoAction
    >>>
    >>> # Automatically detect and create the right environment
    >>> client = AutoEnv.from_docker_image("coding-env:latest")
    >>>
    >>> # Get the corresponding Action class
    >>> CodeAction = AutoAction.from_image("coding-env:latest")
    >>>
    >>> # Use them together
    >>> result = client.reset()
    >>> action = CodeAction(code="print('Hello, AutoEnv!')")
    >>> step_result = client.step(action)
    >>> client.close()
"""

from __future__ import annotations

import importlib
import re
import warnings
from typing import Any, Optional, TYPE_CHECKING

from ._discovery import get_environment_info, discover_environments
from ._registry import get_env_info, list_available_environments

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider
    from core.http_env_client import HTTPEnvClient


class AutoEnv:
    """
    AutoEnv automatically selects and instantiates the correct environment client
    based on Docker image names.

    This class follows the HuggingFace AutoModel pattern, making it easy to work
    with different environments without needing to import specific client classes.

    The class provides factory methods that parse Docker image names, look up the
    corresponding environment in the registry, and return an instance of the
    appropriate client class.

    Example:
        >>> # Simple usage - just specify the image
        >>> env = AutoEnv.from_docker_image("coding-env:latest")
        >>>
        >>> # With custom configuration
        >>> env = AutoEnv.from_docker_image(
        ...     "dipg-env:latest",
        ...     env_vars={"DIPG_DATASET_PATH": "/data/dipg"}
        ... )
        >>>
        >>> # From Hugging Face Hub
        >>> env = AutoEnv.from_hub("openenv/coding-env", tag="v1.0")
        >>>
        >>> # List available environments
        >>> AutoEnv.list_environments()

    Note:
        AutoEnv is not meant to be instantiated directly. Use the class methods
        like from_docker_image() or from_hub() instead.
    """

    def __init__(self):
        """AutoEnv should not be instantiated directly. Use class methods instead."""
        raise TypeError(
            "AutoEnv is a factory class and should not be instantiated directly. "
            "Use AutoEnv.from_docker_image() or AutoEnv.from_hub() instead."
        )

    @classmethod
    def _parse_env_name_from_image(cls, image: str) -> str:
        """
        Extract environment name from Docker image string.

        Supports various image name formats:
        - "coding-env:latest" -> "coding"
        - "ghcr.io/openenv/coding-env:v1.0" -> "coding"
        - "registry.hf.space/org-name-coding-env:latest" -> "coding"

        Args:
            image: Docker image name

        Returns:
            Environment key (e.g., "coding", "atari")

        Raises:
            ValueError: If image name format is invalid
        """
        # Remove registry prefix if present
        # Examples: "ghcr.io/openenv/coding-env:latest", "registry.hf.space/..."
        image_without_registry = re.sub(
            r"^[a-z0-9._-]+\.[a-z]+/", "", image, flags=re.IGNORECASE
        )

        # Remove organization/path prefix if present
        # Example: "openenv/coding-env:latest" -> "coding-env:latest"
        image_without_org = image_without_registry.split("/")[-1]

        # Remove tag if present
        # Example: "coding-env:latest" -> "coding-env"
        image_without_tag = image_without_org.split(":")[0]

        # Extract environment name
        # Pattern: "{env-name}-env" -> "{env-name}"
        # Also support HF format: "org-name-{env-name}-env" -> "{env-name}"
        # First try to match the "-env" suffix pattern
        if image_without_tag.endswith("-env"):
            # Remove the "-env" suffix
            base_name = image_without_tag[:-4]

            # For HF format like "org-name-coding-env", we need the last part before "-env"
            # Split by hyphen and look for known environment names from the end
            parts = base_name.split("-")

            # Try to find a match from the registry starting from the end
            # This handles cases like "openenv-coding" -> "coding"
            for i in range(len(parts)):
                potential_env = "-".join(parts[i:]).replace("-", "_")
                if potential_env in ["sumo_rl"]:  # Special case for underscore envs
                    return potential_env.lower()

                # Check if it could be a valid env name (simple word)
                if i == len(parts) - 1 or len(parts[i:]) == 1:
                    # Last part or single word - likely the env name
                    env_name = parts[-1]
                    return env_name.lower()

            # If we got here, just use the base name
            env_name = base_name
        else:
            # No "-env" suffix, use as-is
            env_name = image_without_tag

        # Clean up: convert underscores as needed
        env_name = env_name.replace("_", "_")  # Keep underscores

        # Validate it looks like an environment name
        if not re.match(r"^[a-z0-9_]+$", env_name, re.IGNORECASE):
            raise ValueError(
                f"Invalid Docker image name format: '{image}'. "
                f"Expected format: '{{env-name}}-env:{{tag}}' or '{{registry}}/{{org}}/{{env-name}}-env:{{tag}}'"
            )

        return env_name.lower()

    @classmethod
    def _get_env_class(cls, env_key: str) -> type:
        """
        Dynamically import and return the environment class.

        Uses auto-discovery first, then falls back to manual registry for
        backward compatibility.

        Args:
            env_key: Environment key (e.g., "coding", "echo")

        Returns:
            Environment class type

        Raises:
            ImportError: If module or class cannot be imported
            ValueError: If environment not found
        """
        # Try auto-discovery first
        discovered_info = get_environment_info(env_key)

        if discovered_info is not None:
            # Use discovered environment
            try:
                return discovered_info.get_client_class()
            except ImportError as e:
                # If discovery finds it but can't load it, raise the error
                raise ImportError(
                    f"Found environment '{env_key}' but failed to load client class: {e}"
                ) from e

        # Fall back to manual registry (deprecated)
        warnings.warn(
            f"Environment '{env_key}' not found via auto-discovery, falling back to "
            f"manual registry. The manual registry is deprecated and will be removed "
            f"in a future version. Please ensure your environment has an openenv.yaml "
            f"manifest or follows the standard directory structure.",
            DeprecationWarning,
            stacklevel=3,
        )

        try:
            env_info = get_env_info(env_key)
        except ValueError:
            # Not in registry either - provide helpful error
            from difflib import get_close_matches

            # Get all available environments from both sources
            discovered = discover_environments()
            suggestions = get_close_matches(
                env_key, list(discovered.keys()), n=3, cutoff=0.6
            )
            suggestion_str = ""
            if suggestions:
                suggestion_str = f" Did you mean: {', '.join(suggestions)}?"

            raise ValueError(
                f"Unknown environment '{env_key}'. "
                f"Available environments: {', '.join(sorted(discovered.keys()))}.{suggestion_str}"
            )

        module_path = env_info["module"]
        class_name = env_info["env_class"]

        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Get the class from the module
            env_class = getattr(module, class_name)

            return env_class

        except ImportError as e:
            raise ImportError(
                f"Failed to import environment module '{module_path}': {e}. "
                f"Make sure the environment package is installed."
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Failed to find class '{class_name}' in module '{module_path}': {e}"
            ) from e

    @classmethod
    def from_docker_image(
        cls,
        image: str,
        provider: Optional["ContainerProvider"] = None,
        wait_timeout: float = 30.0,
        **kwargs: Any,
    ) -> "HTTPEnvClient":
        """
        Create an environment client from a Docker image, automatically detecting
        the environment type.

        This method:
        1. Parses the Docker image name to identify the environment type
        2. Looks up the environment in the registry
        3. Dynamically imports the appropriate client class
        4. Calls that class's from_docker_image() method
        5. Returns the instantiated client

        Args:
            image: Docker image name (e.g., "coding-env:latest")
            provider: Optional container provider (defaults to LocalDockerProvider)
            wait_timeout: Maximum time (in seconds) to wait for container to be ready (default: 30.0)
                         Increase this for slow-starting containers or low-resource environments
            **kwargs: Additional arguments passed to provider.start_container()
                     Common kwargs:
                     - env_vars: Dict of environment variables
                     - port: Port to expose
                     - volumes: Volume mounts

        Returns:
            An instance of the appropriate environment client class

        Raises:
            ValueError: If image name cannot be parsed or environment not found
            ImportError: If environment module cannot be imported
            TimeoutError: If container doesn't become ready within wait_timeout

        Examples:
            >>> # Simple usage
            >>> env = AutoEnv.from_docker_image("coding-env:latest")
            >>> result = env.reset()
            >>> env.close()
            >>>
            >>> # With custom timeout (useful for slow containers)
            >>> env = AutoEnv.from_docker_image(
            ...     "coding-env:latest",
            ...     wait_timeout=60.0  # Wait up to 60 seconds
            ... )
            >>>
            >>> # With environment variables (for DIPG environment)
            >>> env = AutoEnv.from_docker_image(
            ...     "dipg-env:latest",
            ...     wait_timeout=60.0,
            ...     env_vars={"DIPG_DATASET_PATH": "/data/dipg"}
            ... )
            >>>
            >>> # With custom provider
            >>> from core.containers.runtime import LocalDockerProvider
            >>> provider = LocalDockerProvider()
            >>> env = AutoEnv.from_docker_image(
            ...     "coding-env:latest",
            ...     provider=provider,
            ...     wait_timeout=45.0
            ... )
        """
        # Parse environment name from image
        env_key = cls._parse_env_name_from_image(image)

        # Get environment class
        env_class = cls._get_env_class(env_key)

        # Try to get environment info from discovery first for special requirements
        discovered_info = get_environment_info(env_key)

        # If not in discovery, try registry (will be deprecated)
        if discovered_info is None:
            try:
                env_info = get_env_info(env_key)
                special_req = env_info.get("special_requirements")
                if special_req and "env_vars" not in kwargs:
                    warnings.warn(
                        f"Environment '{env_key}' has special requirements: {special_req}. "
                        f"You may need to provide appropriate env_vars.",
                        UserWarning,
                    )
            except ValueError:
                # Not in registry either, no special requirements to warn about
                pass

        # Create and return instance using the class's from_docker_image method
        return env_class.from_docker_image(
            image=image, provider=provider, wait_timeout=wait_timeout, **kwargs
        )

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> "HTTPEnvClient":
        """
        Create an environment client from Hugging Face Hub.

        This is a convenience method that constructs the appropriate Docker image
        name from a Hugging Face repository ID and calls from_docker_image().

        Args:
            repo_id: Hugging Face repository ID (e.g., "openenv/coding-env")
            provider: Optional container provider (defaults to LocalDockerProvider)
            **kwargs: Additional arguments, including:
                     - tag: Docker image tag (default: "latest")
                     - env_vars: Dict of environment variables
                     - Other provider kwargs

        Returns:
            An instance of the appropriate environment client class

        Example:
            >>> # Pull from Hugging Face Hub
            >>> env = AutoEnv.from_hub("openenv/coding-env")
            >>>
            >>> # With specific version
            >>> env = AutoEnv.from_hub("openenv/coding-env", tag="v1.0")
        """
        # Extract tag if provided
        tag = kwargs.pop("tag", "latest")

        # Construct image name for HF registry
        image = f"registry.hf.space/{repo_id.replace('/', '-')}:{tag}"

        # Use from_docker_image with the constructed image name
        return cls.from_docker_image(image=image, provider=provider, **kwargs)

    @classmethod
    def list_environments(cls) -> None:
        """
        Print a list of all available environments with descriptions.

        This is a convenience method for discovering what environments are available.
        Uses auto-discovery to find all environments in the envs directory.

        Example:
            >>> AutoEnv.list_environments()
            Available Environments:
            ----------------------
            atari        : Atari 2600 games environment (100+ games)
            browsergym   : Web browsing environment with multiple benchmarks
            chat         : Chat environment with tokenization support
            ...
        """
        # Use discovery to get all environments
        discovered = discover_environments()

        # Get descriptions from discovered environments
        envs = {key: info.description for key, info in discovered.items()}

        # Fall back to registry if discovery is empty (shouldn't happen normally)
        if not envs:
            warnings.warn(
                "No environments found via auto-discovery, falling back to manual registry. "
                "This is deprecated.",
                DeprecationWarning,
            )
            envs = list_available_environments()

        print("Available Environments:")
        print("-" * 60)

        for env_key in sorted(envs.keys()):
            description = envs[env_key]
            print(f"  {env_key:<15}: {description}")

        print("-" * 60)
        print(f"Total: {len(envs)} environments")
        print("\nUsage:")
        print("  env = AutoEnv.from_docker_image('{env-name}-env:latest')")

    @classmethod
    def from_name(cls, env_name: str) -> type:
        """
        Get the environment class for a specific environment by name.

        This method takes an environment name (key in the registry) and returns
        the corresponding environment class (not an instance).

        Args:
            env_name: Environment name (e.g., "coding", "atari", "echo")

        Returns:
            The environment class for the specified environment (not an instance)

        Raises:
            ValueError: If environment name is not found in registry
            ImportError: If environment class module cannot be imported

        Examples:
            >>> # Get CodingEnv class
            >>> CodingEnv = AutoEnv.from_name("coding")
            >>>
            >>> # Get AtariEnv class
            >>> AtariEnv = AutoEnv.from_name("atari")
            >>>
            >>> # Get EchoEnv class
            >>> EchoEnv = AutoEnv.from_name("echo")
        """
        env_key = env_name.lower()
        return cls._get_env_class(env_key)

    @classmethod
    def get_env_info(cls, env_key: str) -> dict:
        """
        Get detailed information about a specific environment.

        Uses auto-discovery first, then falls back to manual registry.

        Args:
            env_key: Environment key (e.g., "coding", "atari")

        Returns:
            Dictionary with environment information including:
            - description
            - special_requirements (if available)
            - supported_features (if available)
            - default_image
            - version (from discovery)
            - name

        Example:
            >>> info = AutoEnv.get_env_info("coding")
            >>> print(info["description"])
            >>> print(info.get("special_requirements"))
            >>> for feature in info.get("supported_features", []):
            ...     print(f"  - {feature}")
        """
        # Try discovery first
        discovered_info = get_environment_info(env_key)

        if discovered_info is not None:
            # Convert discovered info to dict format
            return {
                "name": discovered_info.name,
                "description": discovered_info.description,
                "version": discovered_info.version,
                "default_image": f"{discovered_info.env_key}-env:latest",
                "env_class": discovered_info.client_class_name,
                "action_class": discovered_info.action_class_name,
                "module": discovered_info.client_module_path,
                # These fields are not available in discovery yet
                "special_requirements": None,
                "supported_features": [],
            }

        # Fall back to registry (deprecated)
        warnings.warn(
            f"Environment '{env_key}' info retrieved from manual registry. "
            f"The manual registry is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_env_info(env_key)
