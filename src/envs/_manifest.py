# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment Manifest Parser
===========================

This module provides functionality to parse and validate openenv.yaml manifest files.
Each environment should contain an openenv.yaml file with metadata about the environment.

Example openenv.yaml:
    spec_version: 1
    name: echo_env
    version: "0.1.0"
    type: space
    runtime: fastapi
    app: server.app:app
    port: 8000

    # AutoEnv metadata (optional)
    client:
      module: client
      class: EchoEnv

    action:
      module: client
      class: EchoAction

    observation:
      module: models
      class: EchoObservation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomli as tomllib  # Python < 3.11
except ImportError:
    try:
        import tomllib  # Python >= 3.11
    except ImportError:
        tomllib = None  # type: ignore

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


@dataclass
class ClientMetadata:
    """Metadata about the client class."""

    module: str
    class_name: str


@dataclass
class ActionMetadata:
    """Metadata about the action class."""

    module: str
    class_name: str


@dataclass
class ObservationMetadata:
    """Metadata about the observation class."""

    module: str
    class_name: str


@dataclass
class EnvironmentManifest:
    """
    Represents the parsed openenv.yaml manifest for an environment.

    Attributes:
        name: Environment name (e.g., "echo_env")
        version: Environment version (e.g., "0.1.0")
        spec_version: Manifest spec version (default: 1)
        type: Environment type (e.g., "space")
        runtime: Runtime type (e.g., "fastapi")
        app: Application entry point (e.g., "server.app:app")
        port: Port number (default: 8000)
        client: Client class metadata
        action: Action class metadata
        observation: Observation class metadata (optional)
        description: Environment description (optional)
        author: Author name (optional)
        license: License type (optional)
        homepage: Homepage URL (optional)
        tags: List of tags (optional)
    """

    name: str
    version: str
    spec_version: int = 1
    type: str = "space"
    runtime: str = "fastapi"
    app: str = "server.app:app"
    port: int = 8000

    client: Optional[ClientMetadata] = None
    action: Optional[ActionMetadata] = None
    observation: Optional[ObservationMetadata] = None

    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    homepage: Optional[str] = None
    tags: Optional[list[str]] = None

    @property
    def env_key(self) -> str:
        """Get the environment key (name without '_env' suffix if present)."""
        name = self.name
        if name.endswith("_env"):
            name = name[:-4]
        return name.lower()


def _infer_class_name_from_env_name(env_name: str, class_type: str) -> str:
    """
    Infer class name from environment name using standard conventions.

    Args:
        env_name: Environment name (e.g., "echo_env", "coding_env")
        class_type: Type of class ("client", "action", "observation")

    Returns:
        Inferred class name

    Examples:
        >>> _infer_class_name_from_env_name("echo_env", "client")
        'EchoEnv'
        >>> _infer_class_name_from_env_name("coding_env", "action")
        'CodeAction'
        >>> _infer_class_name_from_env_name("browsergym_env", "client")
        'BrowserGymEnv'
    """
    # Remove '_env' suffix if present
    base_name = env_name[:-4] if env_name.endswith("_env") else env_name

    # Convert to PascalCase
    # Handle underscores: "browser_gym" -> "BrowserGym"
    parts = base_name.split("_")
    pascal_name = "".join(word.capitalize() for word in parts)

    # Add suffix based on class type
    if class_type == "client":
        return f"{pascal_name}Env"
    elif class_type == "action":
        # Special case: "code" -> "CodeAction" not "CodingAction"
        if base_name == "coding":
            return "CodeAction"
        return f"{pascal_name}Action"
    elif class_type == "observation":
        return f"{pascal_name}Observation"
    else:
        raise ValueError(f"Unknown class type: {class_type}")


def parse_manifest(manifest_path: Path) -> EnvironmentManifest:
    """
    Parse an openenv.yaml manifest file.

    Args:
        manifest_path: Path to openenv.yaml file

    Returns:
        Parsed EnvironmentManifest

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest is invalid
        ImportError: If yaml library is not installed
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to parse openenv.yaml. "
            "Install it with: pip install pyyaml"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # Load YAML
    with open(manifest_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format in {manifest_path}")

    # Required fields
    name = data.get("name")
    if not name:
        raise ValueError(f"Missing required field 'name' in {manifest_path}")

    version = data.get("version", "0.1.0")

    # Optional fields with defaults
    spec_version = data.get("spec_version", 1)
    type_ = data.get("type", "space")
    runtime = data.get("runtime", "fastapi")
    app = data.get("app", "server.app:app")
    port = data.get("port", 8000)

    # Client metadata
    client_data = data.get("client", {})
    client = ClientMetadata(
        module=client_data.get("module", "client"),
        class_name=client_data.get("class", _infer_class_name_from_env_name(name, "client")),
    )

    # Action metadata
    action_data = data.get("action", {})
    action = ActionMetadata(
        module=action_data.get("module", "client"),
        class_name=action_data.get("class", _infer_class_name_from_env_name(name, "action")),
    )

    # Observation metadata (optional)
    observation = None
    if "observation" in data:
        obs_data = data["observation"]
        observation = ObservationMetadata(
            module=obs_data.get("module", "models"),
            class_name=obs_data.get("class", _infer_class_name_from_env_name(name, "observation")),
        )

    # Optional metadata
    description = data.get("description")
    author = data.get("author")
    license_ = data.get("license")
    homepage = data.get("homepage")
    tags = data.get("tags")

    return EnvironmentManifest(
        name=name,
        version=version,
        spec_version=spec_version,
        type=type_,
        runtime=runtime,
        app=app,
        port=port,
        client=client,
        action=action,
        observation=observation,
        description=description,
        author=author,
        license=license_,
        homepage=homepage,
        tags=tags,
    )


def create_manifest_from_convention(env_path: Path) -> EnvironmentManifest:
    """
    Create a manifest by inferring from directory structure and naming conventions.

    This is used for environments that don't have an openenv.yaml file yet.

    Args:
        env_path: Path to environment directory

    Returns:
        Inferred EnvironmentManifest

    Examples:
        >>> manifest = create_manifest_from_convention(Path("src/envs/echo_env"))
        >>> manifest.name
        'echo_env'
        >>> manifest.client.class_name
        'EchoEnv'
    """
    env_name = env_path.name

    # Infer class names from environment name
    client = ClientMetadata(
        module="client",
        class_name=_infer_class_name_from_env_name(env_name, "client"),
    )

    action = ActionMetadata(
        module="client",
        class_name=_infer_class_name_from_env_name(env_name, "action"),
    )

    observation = ObservationMetadata(
        module="models",
        class_name=_infer_class_name_from_env_name(env_name, "observation"),
    )

    # Try to read version from pyproject.toml if available
    version = "0.1.0"
    pyproject_path = env_path / "pyproject.toml"
    if pyproject_path.exists() and tomllib is not None:
        try:
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
                version = pyproject_data.get("project", {}).get("version", "0.1.0")
        except Exception:
            pass  # Use default version

    return EnvironmentManifest(
        name=env_name,
        version=version,
        client=client,
        action=action,
        observation=observation,
        description=f"{env_name} environment",
    )


def load_manifest(env_path: Path) -> EnvironmentManifest:
    """
    Load environment manifest from directory.

    Tries to load from openenv.yaml first, falls back to convention-based inference.

    Args:
        env_path: Path to environment directory

    Returns:
        EnvironmentManifest

    Raises:
        ValueError: If environment directory is invalid
    """
    if not env_path.is_dir():
        raise ValueError(f"Environment path is not a directory: {env_path}")

    manifest_path = env_path / "openenv.yaml"

    if manifest_path.exists():
        # Parse from openenv.yaml
        return parse_manifest(manifest_path)
    else:
        # Infer from conventions
        return create_manifest_from_convention(env_path)
