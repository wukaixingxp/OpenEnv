# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoEnv - Automatic Environment Selection
==========================================

AutoEnv provides a HuggingFace-style API for automatically selecting and
instantiating the correct environment client based on environment names.

This module simplifies environment creation by automatically detecting the
environment type from the name and instantiating the appropriate
client class.

Example:
    >>> from envs import AutoEnv, AutoAction
    >>>
    >>> # Automatically detect and create the right environment
    >>> client = AutoEnv.from_name("coding-env")
    >>>
    >>> # Get the corresponding Action class
    >>> CodeAction = AutoAction.from_name("coding-env")
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

from ._discovery import get_discovery

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider
    from core.http_env_client import HTTPEnvClient


class AutoEnv:
    """
    AutoEnv automatically selects and instantiates the correct environment client
    based on environment names.

    This class follows the HuggingFace AutoModel pattern, making it easy to work
    with different environments without needing to import specific client classes.

    The class provides factory methods that parse environment names, look up the
    corresponding environment in the registry, and return an instance of the
    appropriate client class.

    Example:
        >>> # Simple usage - just specify the name
        >>> env = AutoEnv.from_name("coding-env")
        >>>
        >>> # With custom configuration
        >>> env = AutoEnv.from_name(
        ...     "dipg-env",
        ...     env_vars={"DIPG_DATASET_PATH": "/data/dipg"}
        ... )
        >>>
        >>> # List available environments
        >>> AutoEnv.list_environments()

    Note:
        AutoEnv is not meant to be instantiated directly. Use the class method
        from_name() instead.
    """

    def __init__(self):
        """AutoEnv should not be instantiated directly. Use class methods instead."""
        raise TypeError(
            "AutoEnv is a factory class and should not be instantiated directly. "
            "Use AutoEnv.from_name() instead."
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

        Uses auto-discovery to find and load the environment class.

        Args:
            env_key: Environment key (e.g., "coding", "echo")

        Returns:
            Environment class type

        Raises:
            ImportError: If module or class cannot be imported
            ValueError: If environment not found
        """
        # Use discovery to find environment
        discovery = get_discovery()
        env_info = discovery.get_environment(env_key)

        if env_info is None:
            # Try to suggest similar environment names
            from difflib import get_close_matches

            all_envs = discovery.discover()
            suggestions = get_close_matches(env_key, all_envs.keys(), n=3, cutoff=0.6)
            suggestion_str = ""
            if suggestions:
                suggestion_str = f" Did you mean: {', '.join(suggestions)}?"

            raise ValueError(
                f"Unknown environment '{env_key}'. "
                f"Supported environments: {', '.join(sorted(all_envs.keys()))}.{suggestion_str}"
            )

        # Import and return the client class
        try:
            return env_info.get_client_class()
        except ImportError as e:
            raise ImportError(
                f"Failed to import {env_info.client_class_name} from {env_info.client_module_path}: {e}. "
                f"Make sure the environment package is installed."
            ) from e

    @classmethod
    def from_name(
        cls,
        name: str,
        provider: Optional["ContainerProvider"] = None,
        wait_timeout: float = 30.0,
        **kwargs: Any,
    ) -> "HTTPEnvClient":
        """
        Create an environment client from an environment name, automatically detecting
        the environment type and handling Docker image details.

        This method:
        1. Parses the environment name to identify the environment type
        2. Looks up the environment in the registry
        3. Dynamically imports the appropriate client class
        4. Calls that class's from_docker_image() method with the appropriate image
        5. Returns the instantiated client

        Args:
            name: Environment name (e.g., "coding-env", "coding-env:latest", or "coding")
                  If no tag is provided, ":latest" is automatically appended
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
            ValueError: If name cannot be parsed or environment not found
            ImportError: If environment module cannot be imported
            TimeoutError: If container doesn't become ready within wait_timeout

        Examples:
            >>> # Simple usage with environment name
            >>> env = AutoEnv.from_name("coding-env")
            >>> result = env.reset()
            >>> env.close()
            >>>
            >>> # With tag specified
            >>> env = AutoEnv.from_name("coding-env:v1.0")
            >>>
            >>> # With custom timeout (useful for slow containers)
            >>> env = AutoEnv.from_name(
            ...     "coding-env",
            ...     wait_timeout=60.0  # Wait up to 60 seconds
            ... )
            >>>
            >>> # With environment variables (for DIPG environment)
            >>> env = AutoEnv.from_name(
            ...     "dipg-env",
            ...     wait_timeout=60.0,
            ...     env_vars={"DIPG_DATASET_PATH": "/data/dipg"}
            ... )
            >>>
            >>> # With custom provider
            >>> from core.containers.runtime import LocalDockerProvider
            >>> provider = LocalDockerProvider()
            >>> env = AutoEnv.from_name(
            ...     "coding-env",
            ...     provider=provider,
            ...     wait_timeout=45.0
            ... )
        """
        # Normalize name to image format
        # If name doesn't have a tag and doesn't end with -env, add -env suffix
        # If name has -env but no tag, add :latest
        image = name
        if ":" not in name:
            # No tag provided, add :latest
            if not name.endswith("-env"):
                # Name is like "coding", convert to "coding-env:latest"
                image = f"{name}-env:latest"
            else:
                # Name is like "coding-env", add :latest
                image = f"{name}:latest"
        elif not name.split(":")[0].endswith("-env"):
            # Has tag but no -env suffix, add -env
            # e.g., "coding:v1.0" -> "coding-env:v1.0"
            base, tag = name.split(":", 1)
            image = f"{base}-env:{tag}"

        # Parse environment name from image
        env_key = cls._parse_env_name_from_image(image)

        # Get environment class
        env_class = cls._get_env_class(env_key)

        # Create and return instance using the class's from_docker_image method
        return env_class.from_docker_image(
            image=image, provider=provider, wait_timeout=wait_timeout, **kwargs
        )

    @classmethod
    def list_environments(cls) -> None:
        """
        Print a list of all available environments with descriptions.

        Uses auto-discovery to find all environments.

        Example:
            >>> AutoEnv.list_environments()
            Available Environments:
            ----------------------------------------------------------------------
            atari          : Atari Env environment (v0.1.0)
            browsergym     : Browsergym Env environment (v0.1.0)
            coding         : Coding Env environment (v0.1.0)
            ...
        """
        # Use discovery
        discovery = get_discovery()
        discovered_envs = discovery.discover()

        if discovered_envs:
            print("Available Environments:")
            print("-" * 70)

            for env_key in sorted(discovered_envs.keys()):
                env = discovered_envs[env_key]
                print(f"  {env_key:<15}: {env.description} (v{env.version})")

            print("-" * 70)
            print(f"Total: {len(discovered_envs)} environments")
            print("\nUsage:")
            print("  env = AutoEnv.from_name('coding-env')")
        else:
            print("No environments found.")
            print("Make sure your environments are in the src/envs/ directory.")
            print("Each environment should have either:")
            print("  - An openenv.yaml manifest file")
            print("  - Or follow the standard directory structure with client.py")

    @classmethod
    def get_env_class(cls, env_name: str) -> type:
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
            >>> CodingEnv = AutoEnv.get_env_class("coding")
            >>>
            >>> # Get AtariEnv class
            >>> AtariEnv = AutoEnv.get_env_class("atari")
            >>>
            >>> # Get EchoEnv class
            >>> EchoEnv = AutoEnv.get_env_class("echo")
        """
        env_key = env_name.lower()
        return cls._get_env_class(env_key)

    @classmethod
    def get_env_info(cls, env_key: str) -> dict:
        """
        Get detailed information about a specific environment.

        Uses auto-discovery to find environment information.

        Args:
            env_key: Environment key (e.g., "coding", "atari")

        Returns:
            Dictionary with environment information including:
            - name
            - description
            - version
            - default_image
            - env_class
            - action_class
            - observation_class
            - module
            - spec_version

        Raises:
            ValueError: If environment not found

        Example:
            >>> info = AutoEnv.get_env_info("coding")
            >>> print(info["description"])
            >>> print(info["version"])
            >>> print(info["default_image"])
        """
        # Use discovery
        discovery = get_discovery()
        env_info = discovery.get_environment(env_key)

        if env_info is None:
            raise ValueError(
                f"Environment '{env_key}' not found. Use AutoEnv.list_environments() "
                f"to see all available environments."
            )

        # Return info from discovery
        return {
            "name": env_info.name,
            "description": env_info.description,
            "version": env_info.version,
            "default_image": env_info.default_image,
            "env_class": env_info.client_class_name,
            "action_class": env_info.action_class_name,
            "observation_class": env_info.observation_class_name,
            "module": env_info.client_module_path,
            "spec_version": env_info.spec_version,
        }
