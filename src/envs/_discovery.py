# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment Auto-Discovery System
==================================

This module provides automatic discovery of OpenEnv environments by:
1. Scanning the src/envs/ directory for environment directories
2. Loading manifests (from openenv.yaml or conventions)
3. Caching results for performance

This enables AutoEnv to work without a manual registry.
"""

import importlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

from ._manifest import load_manifest, EnvironmentManifest

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """
    Rich information about a discovered environment.

    Attributes:
        env_key: Environment key (e.g., "echo", "coding")
        name: Full environment name (e.g., "echo_env")
        version: Version string
        description: Human-readable description
        env_dir: Path to environment directory
        client_module_path: Full module path to client (e.g., "envs.echo_env.client")
        action_module_path: Full module path to action module
        observation_module_path: Full module path to observation module
        client_class_name: Client class name (e.g., "EchoEnv")
        action_class_name: Action class name (e.g., "EchoAction")
        observation_class_name: Observation class name
        default_image: Default Docker image name (e.g., "echo-env:latest")
        spec_version: OpenEnv spec version (from openenv.yaml)
        manifest: Original manifest data
    """
    env_key: str
    name: str
    version: str
    description: str
    env_dir: str
    client_module_path: str
    action_module_path: str
    observation_module_path: str
    client_class_name: str
    action_class_name: str
    observation_class_name: str
    default_image: str
    spec_version: Optional[int] = None
    manifest: Optional[Dict[str, Any]] = None

    def get_client_class(self) -> Type:
        """
        Dynamically import and return the client class.

        Returns:
            Client class (e.g., EchoEnv)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.client_module_path)
            return getattr(module, self.client_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.client_class_name} from {self.client_module_path}: {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.client_class_name} not found in {self.client_module_path}: {e}"
            ) from e

    def get_action_class(self) -> Type:
        """
        Dynamically import and return the action class.

        Returns:
            Action class (e.g., EchoAction)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.action_module_path)
            return getattr(module, self.action_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.action_class_name} from {self.action_module_path}: {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.action_class_name} not found in {self.action_module_path}: {e}"
            ) from e

    def get_observation_class(self) -> Type:
        """
        Dynamically import and return the observation class.

        Returns:
            Observation class (e.g., EchoObservation)

        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module = importlib.import_module(self.observation_module_path)
            return getattr(module, self.observation_class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {self.observation_class_name} from {self.observation_module_path}: {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class {self.observation_class_name} not found in {self.observation_module_path}: {e}"
            ) from e


class EnvironmentDiscovery:
    """
    Auto-discovery system for OpenEnv environments.

    This class scans a directory for environments, loads their manifests,
    and caches the results for performance.
    """

    def __init__(self, envs_dir: Path, module_prefix: str = "envs"):
        """
        Initialize discovery system.

        Args:
            envs_dir: Directory containing environments (e.g., /path/to/src/envs)
            module_prefix: Module prefix for imports (e.g., "envs")
        """
        self.envs_dir = Path(envs_dir)
        self.module_prefix = module_prefix
        self._cache_file = self.envs_dir / ".discovery_cache.json"
        self._cache: Optional[Dict[str, EnvironmentInfo]] = None

    def _is_valid_env_dir(self, dir_path: Path) -> bool:
        """
        Check if a directory is a valid environment directory.

        A directory is considered valid if it:
        - Is a directory (not a file)
        - Doesn't start with . or _
        - Contains client.py or server/ subdirectory

        Args:
            dir_path: Path to check

        Returns:
            True if valid environment directory
        """
        if not dir_path.is_dir():
            return False

        # Skip hidden directories and special directories
        if dir_path.name.startswith(".") or dir_path.name.startswith("_"):
            return False

        # Check for client.py or server/ directory
        has_client = (dir_path / "client.py").exists()
        has_server = (dir_path / "server").is_dir()

        return has_client or has_server

    def _create_env_info(self, manifest: EnvironmentManifest, env_dir: Path) -> EnvironmentInfo:
        """
        Create EnvironmentInfo from a manifest.

        Args:
            manifest: Parsed environment manifest
            env_dir: Path to environment directory

        Returns:
            EnvironmentInfo instance
        """
        # Determine env_key (e.g., "echo_env" → "echo")
        env_key = manifest.name.replace("_env", "") if manifest.name.endswith("_env") else manifest.name

        # Construct module paths
        # e.g., "envs.echo_env.client"
        client_module_path = f"{self.module_prefix}.{manifest.name}.{manifest.client.module}"
        action_module_path = f"{self.module_prefix}.{manifest.name}.{manifest.action.module}"
        observation_module_path = f"{self.module_prefix}.{manifest.name}.{manifest.observation.module}"

        # Determine default Docker image name
        # e.g., "echo_env" → "echo-env:latest"
        image_name = manifest.name.replace("_", "-")
        default_image = f"{image_name}:latest"

        return EnvironmentInfo(
            env_key=env_key,
            name=manifest.name,
            version=manifest.version,
            description=manifest.description,
            env_dir=str(env_dir),
            client_module_path=client_module_path,
            action_module_path=action_module_path,
            observation_module_path=observation_module_path,
            client_class_name=manifest.client.class_name,
            action_class_name=manifest.action.class_name,
            observation_class_name=manifest.observation.class_name,
            default_image=default_image,
            spec_version=manifest.spec_version,
            manifest=manifest.raw_data
        )

    def _load_cache(self) -> Optional[Dict[str, EnvironmentInfo]]:
        """
        Load cached discovery results.

        Returns:
            Dictionary of env_key -> EnvironmentInfo, or None if cache invalid
        """
        if not self._cache_file.exists():
            return None

        try:
            with open(self._cache_file, "r") as f:
                cache_data = json.load(f)

            # Reconstruct EnvironmentInfo objects
            cache = {}
            for env_key, env_data in cache_data.items():
                cache[env_key] = EnvironmentInfo(**env_data)

            return cache
        except Exception as e:
            logger.warning(f"Failed to load discovery cache: {e}")
            return None

    def _save_cache(self, environments: Dict[str, EnvironmentInfo]) -> None:
        """
        Save discovery results to cache.

        Args:
            environments: Dictionary of env_key -> EnvironmentInfo
        """
        try:
            cache_data = {}
            for env_key, env_info in environments.items():
                cache_data[env_key] = asdict(env_info)

            with open(self._cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save discovery cache: {e}")

    def discover(self, use_cache: bool = True) -> Dict[str, EnvironmentInfo]:
        """
        Discover all environments in the envs directory.

        Args:
            use_cache: If True, try to load from cache first

        Returns:
            Dictionary mapping env_key to EnvironmentInfo

        Examples:
            >>> discovery = EnvironmentDiscovery(Path("src/envs"))
            >>> envs = discovery.discover()
            >>> print(envs.keys())
            dict_keys(['echo', 'coding', 'atari', ...])
        """
        # Try to load from cache first
        if use_cache and self._cache is not None:
            return self._cache

        if use_cache:
            cached = self._load_cache()
            if cached is not None:
                self._cache = cached
                return self._cache

        # Scan directory for environments
        environments = {}

        if not self.envs_dir.exists():
            logger.warning(f"Environments directory not found: {self.envs_dir}")
            return environments

        for item in self.envs_dir.iterdir():
            if not self._is_valid_env_dir(item):
                continue

            try:
                # Load manifest (from openenv.yaml or conventions)
                manifest = load_manifest(item)

                # Create environment info
                env_info = self._create_env_info(manifest, item)

                # Add to discovered environments
                environments[env_info.env_key] = env_info

                logger.debug(f"Discovered environment: {env_info.env_key}")

            except Exception as e:
                logger.warning(f"Failed to load environment from {item}: {e}")
                continue

        # Save to cache
        self._save_cache(environments)
        self._cache = environments

        return environments

    def get_environment(self, env_key: str) -> Optional[EnvironmentInfo]:
        """
        Get information about a specific environment.

        Args:
            env_key: Environment key (e.g., "echo", "coding")

        Returns:
            EnvironmentInfo if found, None otherwise

        Examples:
            >>> discovery = EnvironmentDiscovery(Path("src/envs"))
            >>> env = discovery.get_environment("echo")
            >>> print(env.client_class_name)
            'EchoEnv'
        """
        environments = self.discover()
        return environments.get(env_key)

    def list_environments(self) -> None:
        """
        Print a formatted list of all discovered environments.

        Examples:
            >>> discovery = EnvironmentDiscovery(Path("src/envs"))
            >>> discovery.list_environments()
            Discovered Environments:
            ----------------------------------------------------------------------
              echo           : Echo Env environment (v0.1.0)
              coding         : Coding Env environment (v0.1.0)
              ...
        """
        environments = self.discover()

        print("Discovered Environments:")
        print("-" * 70)

        for env_key in sorted(environments.keys()):
            env = environments[env_key]
            print(f"  {env_key:<15}: {env.description} (v{env.version})")

        print("-" * 70)
        print(f"Total: {len(environments)} environments")

    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        if self._cache_file.exists():
            self._cache_file.unlink()
        self._cache = None


# Global discovery instance
_global_discovery: Optional[EnvironmentDiscovery] = None


def get_discovery(envs_dir: Optional[Path] = None, module_prefix: str = "envs") -> EnvironmentDiscovery:
    """
    Get or create the global discovery instance.

    Args:
        envs_dir: Directory containing environments (default: src/envs relative to this file)
        module_prefix: Module prefix for imports (default: "envs")

    Returns:
        Global EnvironmentDiscovery instance

    Examples:
        >>> discovery = get_discovery()
        >>> envs = discovery.discover()
    """
    global _global_discovery

    if _global_discovery is None:
        if envs_dir is None:
            # Default to src/envs relative to this file
            # This file is in src/envs/_discovery.py
            # So parent is src/envs/
            envs_dir = Path(__file__).parent

        _global_discovery = EnvironmentDiscovery(envs_dir, module_prefix)

    return _global_discovery


def reset_discovery() -> None:
    """Reset the global discovery instance (useful for testing)."""
    global _global_discovery
    _global_discovery = None
