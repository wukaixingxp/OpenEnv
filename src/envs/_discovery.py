# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment Auto-Discovery
==========================

This module provides functionality to automatically discover available environments
by scanning directories for environment manifests and client modules.

This replaces the manual registry approach with dynamic discovery.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ._manifest import EnvironmentManifest, load_manifest


class EnvironmentInfo:
    """Information about a discovered environment."""

    def __init__(
        self,
        env_key: str,
        manifest: EnvironmentManifest,
        env_path: Path,
        module_prefix: str = "envs",
    ):
        """
        Initialize environment info.

        Args:
            env_key: Environment key (e.g., "echo", "coding")
            manifest: Environment manifest
            env_path: Path to environment directory
            module_prefix: Python module prefix for importing (e.g., "envs")
        """
        self.env_key = env_key
        self.manifest = manifest
        self.env_path = env_path
        self.module_prefix = module_prefix

    @property
    def name(self) -> str:
        """Environment name from manifest."""
        return self.manifest.name

    @property
    def version(self) -> str:
        """Environment version."""
        return self.manifest.version

    @property
    def description(self) -> str:
        """Environment description."""
        return self.manifest.description or f"{self.name} environment"

    @property
    def client_module_path(self) -> str:
        """Full module path to client module (e.g., 'envs.echo_env.client')."""
        return f"{self.module_prefix}.{self.name}.{self.manifest.client.module}"

    @property
    def client_class_name(self) -> str:
        """Client class name (e.g., 'EchoEnv')."""
        return self.manifest.client.class_name

    @property
    def action_module_path(self) -> str:
        """Full module path to action module."""
        return f"{self.module_prefix}.{self.name}.{self.manifest.action.module}"

    @property
    def action_class_name(self) -> str:
        """Action class name (e.g., 'EchoAction')."""
        return self.manifest.action.class_name

    @property
    def observation_module_path(self) -> Optional[str]:
        """Full module path to observation module (if specified)."""
        if self.manifest.observation:
            return f"{self.module_prefix}.{self.name}.{self.manifest.observation.module}"
        return None

    @property
    def observation_class_name(self) -> Optional[str]:
        """Observation class name (if specified)."""
        if self.manifest.observation:
            return self.manifest.observation.class_name
        return None

    def get_client_class(self) -> type:
        """
        Dynamically import and return the client class.

        Returns:
            Client class type

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.client_module_path)
            return getattr(module, self.client_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import client module '{self.client_module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Failed to find client class '{self.client_class_name}' "
                f"in module '{self.client_module_path}': {e}"
            ) from e

    def get_action_class(self) -> type:
        """
        Dynamically import and return the action class.

        Returns:
            Action class type

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.action_module_path)
            return getattr(module, self.action_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import action module '{self.action_module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Failed to find action class '{self.action_class_name}' "
                f"in module '{self.action_module_path}': {e}"
            ) from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "env_key": self.env_key,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "client_module": self.client_module_path,
            "client_class": self.client_class_name,
            "action_module": self.action_module_path,
            "action_class": self.action_class_name,
            "env_path": str(self.env_path),
        }


class EnvironmentDiscovery:
    """
    Discovers and manages available environments.

    This class scans directories for environments and caches the results for performance.
    """

    def __init__(self, envs_dir: Optional[Path] = None, module_prefix: str = "envs"):
        """
        Initialize environment discovery.

        Args:
            envs_dir: Directory to scan for environments (default: src/envs)
            module_prefix: Python module prefix for imports (default: "envs")
        """
        if envs_dir is None:
            # Auto-detect envs directory
            current_file = Path(__file__)
            envs_dir = current_file.parent  # This is src/envs/

        self.envs_dir = envs_dir
        self.module_prefix = module_prefix
        self._cache: Optional[Dict[str, EnvironmentInfo]] = None
        self._cache_file = self.envs_dir / ".discovery_cache.json"

    def discover(self, force_refresh: bool = False) -> Dict[str, EnvironmentInfo]:
        """
        Discover all available environments.

        Args:
            force_refresh: Force re-scanning even if cache exists

        Returns:
            Dictionary mapping env_key to EnvironmentInfo
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        # Try to load from cache file
        if not force_refresh and self._load_from_cache():
            return self._cache

        # Scan directory
        environments = {}

        if not self.envs_dir.exists():
            self._cache = environments
            return environments

        # Scan all subdirectories
        for env_dir in self.envs_dir.iterdir():
            # Skip non-directories and special files
            if not env_dir.is_dir():
                continue
            if env_dir.name.startswith(".") or env_dir.name.startswith("_"):
                continue
            if env_dir.name == "__pycache__":
                continue

            # Check if this looks like an environment (has client.py or openenv.yaml)
            client_file = env_dir / "client.py"
            manifest_file = env_dir / "openenv.yaml"

            if not (client_file.exists() or manifest_file.exists()):
                continue

            try:
                # Load manifest (from yaml or infer from conventions)
                manifest = load_manifest(env_dir)

                # Create environment info
                env_info = EnvironmentInfo(
                    env_key=manifest.env_key,
                    manifest=manifest,
                    env_path=env_dir,
                    module_prefix=self.module_prefix,
                )

                environments[env_info.env_key] = env_info

            except Exception as e:
                # Log warning but continue discovery
                import warnings

                warnings.warn(
                    f"Failed to load environment from {env_dir}: {e}",
                    UserWarning,
                )
                continue

        self._cache = environments
        self._save_to_cache()

        return environments

    def get_environment(self, env_key: str) -> Optional[EnvironmentInfo]:
        """
        Get information about a specific environment.

        Args:
            env_key: Environment key (e.g., "echo", "coding")

        Returns:
            EnvironmentInfo if found, None otherwise
        """
        environments = self.discover()
        return environments.get(env_key.lower())

    def list_environments(self) -> Dict[str, str]:
        """
        List all available environments with descriptions.

        Returns:
            Dictionary mapping env_key to description
        """
        environments = self.discover()
        return {key: info.description for key, info in environments.items()}

    def _load_from_cache(self) -> bool:
        """
        Load discovery results from cache file.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not self._cache_file.exists():
            return False

        try:
            # Check if cache is stale (older than any environment directory)
            cache_mtime = self._cache_file.stat().st_mtime

            # If any env directory is newer than cache, invalidate
            for env_dir in self.envs_dir.iterdir():
                if not env_dir.is_dir() or env_dir.name.startswith((".", "_")):
                    continue
                if env_dir.stat().st_mtime > cache_mtime:
                    return False

            # Load cache
            with open(self._cache_file, "r") as f:
                cache_data = json.load(f)

            # Reconstruct EnvironmentInfo objects
            environments = {}
            for env_key, data in cache_data.items():
                env_path = Path(data["env_path"])
                if not env_path.exists():
                    continue

                manifest = load_manifest(env_path)
                env_info = EnvironmentInfo(
                    env_key=env_key,
                    manifest=manifest,
                    env_path=env_path,
                    module_prefix=self.module_prefix,
                )
                environments[env_key] = env_info

            self._cache = environments
            return True

        except Exception:
            # If cache loading fails, just re-scan
            return False

    def _save_to_cache(self) -> None:
        """Save discovery results to cache file."""
        if self._cache is None:
            return

        try:
            cache_data = {key: info.to_dict() for key, info in self._cache.items()}

            with open(self._cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

        except Exception:
            # Cache save is optional, so just ignore errors
            pass


# Global discovery instance
_global_discovery: Optional[EnvironmentDiscovery] = None


def get_discovery() -> EnvironmentDiscovery:
    """
    Get the global environment discovery instance.

    Returns:
        EnvironmentDiscovery instance
    """
    global _global_discovery
    if _global_discovery is None:
        _global_discovery = EnvironmentDiscovery()
    return _global_discovery


def discover_environments(force_refresh: bool = False) -> Dict[str, EnvironmentInfo]:
    """
    Discover all available environments (convenience function).

    Args:
        force_refresh: Force re-scanning even if cache exists

    Returns:
        Dictionary mapping env_key to EnvironmentInfo
    """
    return get_discovery().discover(force_refresh=force_refresh)


def get_environment_info(env_key: str) -> Optional[EnvironmentInfo]:
    """
    Get information about a specific environment (convenience function).

    Args:
        env_key: Environment key (e.g., "echo", "coding")

    Returns:
        EnvironmentInfo if found, None otherwise
    """
    return get_discovery().get_environment(env_key)
