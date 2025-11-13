# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoAction - Automatic Action Class Selection
==============================================

AutoAction provides a HuggingFace-style API for automatically retrieving the
correct Action class based on environment names or Docker image names.

This module simplifies working with environment actions by automatically
detecting and returning the appropriate Action class without requiring
manual imports.

Example:
    >>> from envs import AutoEnv, AutoAction
    >>>
    >>> # Get Action class from environment name
    >>> CodeAction = AutoAction.from_env("coding")
    >>>
    >>> # Or get Action class from Docker image
    >>> CodeAction = AutoAction.from_image("coding-env:latest")
    >>>
    >>> # Use the Action class
    >>> action = CodeAction(code="print('Hello!')")
    >>>
    >>> # Use with AutoEnv
    >>> env = AutoEnv.from_docker_image("coding-env:latest")
    >>> result = env.step(action)
"""

from __future__ import annotations

import importlib
import re
import warnings
from typing import Type

from ._discovery import get_environment_info, discover_environments
from ._registry import get_env_info


class AutoAction:
    """
    AutoAction automatically retrieves the correct Action class based on
    environment names or Docker image names.

    This class follows the HuggingFace AutoModel pattern, making it easy to
    get the right Action class without needing to know which module to import.

    The class provides factory methods that look up the Action class in the
    registry and return the class (not an instance) for you to instantiate.

    Example:
        >>> # Get Action class from environment name
        >>> CodeAction = AutoAction.from_env("coding")
        >>> action = CodeAction(code="print('test')")
        >>>
        >>> # Get Action class from Docker image name
        >>> CodeAction = AutoAction.from_image("coding-env:latest")
        >>> action = CodeAction(code="print('test')")
        >>>
        >>> # Use with AutoEnv for a complete workflow
        >>> env = AutoEnv.from_docker_image("coding-env:latest")
        >>> ActionClass = AutoAction.from_image("coding-env:latest")
        >>> action = ActionClass(code="print('Hello, AutoAction!')")
        >>> result = env.step(action)

    Note:
        AutoAction is not meant to be instantiated directly. Use the class
        methods like from_env() or from_image() instead.
    """

    def __init__(self):
        """AutoAction should not be instantiated directly. Use class methods instead."""
        raise TypeError(
            "AutoAction is a factory class and should not be instantiated directly. "
            "Use AutoAction.from_env() or AutoAction.from_image() instead."
        )

    @classmethod
    def _parse_env_name_from_image(cls, image: str) -> str:
        """
        Extract environment name from Docker image string.

        This method uses the same parsing logic as AutoEnv to ensure consistency.

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
        image_without_registry = re.sub(r"^[a-z0-9._-]+\.[a-z]+/", "", image, flags=re.IGNORECASE)

        # Remove organization/path prefix if present
        image_without_org = image_without_registry.split("/")[-1]

        # Remove tag if present
        image_without_tag = image_without_org.split(":")[0]

        # Extract environment name
        # Pattern: "{env-name}-env" -> "{env-name}"
        # Also support HF format: "org-name-{env-name}-env" -> "{env-name}"
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
        
        # Clean up: keep underscores
        env_name = env_name.replace("_", "_")
        
        # Validate it looks like an environment name
        if not re.match(r"^[a-z0-9_]+$", env_name, re.IGNORECASE):
            raise ValueError(
                f"Invalid Docker image name format: '{image}'. "
                f"Expected format: '{{env-name}}-env:{{tag}}' or '{{registry}}/{{org}}/{{env-name}}-env:{{tag}}'"
            )

        return env_name.lower()

    @classmethod
    def _get_action_class(cls, env_key: str) -> Type:
        """
        Dynamically import and return the Action class for an environment.

        Uses auto-discovery first, then falls back to manual registry for
        backward compatibility.

        Args:
            env_key: Environment key (e.g., "coding", "atari")

        Returns:
            Action class type (not an instance)

        Raises:
            ImportError: If module or class cannot be imported
            ValueError: If environment not found
        """
        # Try auto-discovery first
        discovered_info = get_environment_info(env_key)

        if discovered_info is not None:
            # Use discovered environment
            try:
                return discovered_info.get_action_class()
            except ImportError as e:
                # If discovery finds it but can't load it, raise the error
                raise ImportError(
                    f"Found environment '{env_key}' but failed to load action class: {e}"
                ) from e

        # Fall back to manual registry (deprecated)
        warnings.warn(
            f"Action for environment '{env_key}' not found via auto-discovery, "
            f"falling back to manual registry. The manual registry is deprecated and "
            f"will be removed in a future version.",
            DeprecationWarning,
            stacklevel=3,
        )

        try:
            env_info = get_env_info(env_key)
        except ValueError:
            # Not in registry either - provide helpful error
            from difflib import get_close_matches

            # Get all available environments from discovery
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
        action_class_name = env_info["action_class"]

        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Get the Action class from the module
            action_class = getattr(module, action_class_name)

            return action_class

        except ImportError as e:
            raise ImportError(
                f"Failed to import environment module '{module_path}': {e}. "
                f"Make sure the environment package is installed."
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Failed to find Action class '{action_class_name}' in module '{module_path}': {e}"
            ) from e

    @classmethod
    def from_env(cls, env_name: str) -> Type:
        """
        Get the Action class for a specific environment by name.

        This method takes an environment name (key in the registry) and returns
        the corresponding Action class.

        Args:
            env_name: Environment name (e.g., "coding", "atari", "echo")

        Returns:
            The Action class for the specified environment (not an instance)

        Raises:
            ValueError: If environment name is not found in registry
            ImportError: If Action class module cannot be imported

        Examples:
            >>> # Get CodeAction class
            >>> CodeAction = AutoAction.from_env("coding")
            >>> action = CodeAction(code="print('Hello!')")
            >>>
            >>> # Get AtariAction class
            >>> AtariAction = AutoAction.from_env("atari")
            >>> action = AtariAction(action=0)  # Fire button
            >>>
            >>> # Get EchoAction class
            >>> EchoAction = AutoAction.from_env("echo")
            >>> action = EchoAction(message="Hello!")
        """
        env_key = env_name.lower()
        return cls._get_action_class(env_key)

    @classmethod
    def from_image(cls, image: str) -> Type:
        """
        Get the Action class for an environment by parsing its Docker image name.

        This method takes a Docker image name, extracts the environment type,
        and returns the corresponding Action class.

        Args:
            image: Docker image name (e.g., "coding-env:latest")

        Returns:
            The Action class for the environment (not an instance)

        Raises:
            ValueError: If image name cannot be parsed or environment not found
            ImportError: If Action class module cannot be imported

        Examples:
            >>> # Get CodeAction from image name
            >>> CodeAction = AutoAction.from_image("coding-env:latest")
            >>> action = CodeAction(code="print('Hello!')")
            >>>
            >>> # With full registry path
            >>> CodeAction = AutoAction.from_image("ghcr.io/openenv/coding-env:v1.0")
            >>> action = CodeAction(code="x = 5 + 3")
            >>>
            >>> # From Hugging Face Hub format
            >>> CodeAction = AutoAction.from_image("registry.hf.space/openenv-coding-env:latest")
            >>> action = CodeAction(code="import math")
        """
        env_key = cls._parse_env_name_from_image(image)
        return cls._get_action_class(env_key)

    @classmethod
    def get_action_info(cls, env_name: str) -> dict:
        """
        Get information about the Action class for an environment.

        Uses auto-discovery first, then falls back to manual registry.

        Args:
            env_name: Environment name (e.g., "coding", "atari")

        Returns:
            Dictionary with Action class information including module and class name

        Example:
            >>> info = AutoAction.get_action_info("coding")
            >>> print(info["action_class"])  # "CodeAction"
            >>> print(info["module"])  # "envs.coding_env.client"
        """
        env_key = env_name.lower()

        # Try discovery first
        discovered_info = get_environment_info(env_key)

        if discovered_info is not None:
            return {
                "action_class": discovered_info.action_class_name,
                "module": discovered_info.action_module_path,
                "env_class": discovered_info.client_class_name,
                "description": discovered_info.description,
            }

        # Fall back to registry (deprecated)
        warnings.warn(
            f"Action info for environment '{env_key}' retrieved from manual registry. "
            f"The manual registry is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )

        env_info = get_env_info(env_key)

        return {
            "action_class": env_info["action_class"],
            "module": env_info["module"],
            "env_class": env_info["env_class"],
            "description": env_info["description"],
        }

    @classmethod
    def list_actions(cls) -> None:
        """
        Print a list of all available Action classes.

        Uses auto-discovery to find all environments and their action classes.

        Example:
            >>> AutoAction.list_actions()
            Available Action Classes:
            -------------------------
            coding       : CodeAction       (Python code execution environment)
            atari        : AtariAction      (Atari 2600 games environment (100+ games))
            echo         : EchoAction       (Simple echo test environment)
            ...
        """
        # Use discovery to get all environments
        discovered = discover_environments()

        # Fall back to registry if discovery is empty
        if not discovered:
            warnings.warn(
                "No environments found via auto-discovery, falling back to manual registry. "
                "This is deprecated.",
                DeprecationWarning,
            )
            from ._registry import ENV_REGISTRY

            print("Available Action Classes:")
            print("-" * 70)

            for env_key in sorted(ENV_REGISTRY.keys()):
                info = ENV_REGISTRY[env_key]
                action_class = info["action_class"]
                description = info["description"]
                print(f"  {env_key:<15}: {action_class:<20} ({description})")

            print("-" * 70)
            print(f"Total: {len(ENV_REGISTRY)} Action classes")
        else:
            print("Available Action Classes:")
            print("-" * 70)

            for env_key in sorted(discovered.keys()):
                info = discovered[env_key]
                action_class = info.action_class_name
                description = info.description
                print(f"  {env_key:<15}: {action_class:<20} ({description})")

            print("-" * 70)
            print(f"Total: {len(discovered)} Action classes")

        print("\nUsage:")
        print("  ActionClass = AutoAction.from_env('env-name')")
        print("  # or")
        print("  ActionClass = AutoAction.from_image('env-name-env:latest')")
