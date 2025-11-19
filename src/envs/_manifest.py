# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment Manifest Parser
============================

This module provides functionality to parse environment metadata from:
1. openenv.yaml manifest files (if they exist)
2. Convention-based inference from directory structure

The parser supports both PR #160 format and custom metadata extensions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class ClientMetadata:
    """Metadata about the environment client class."""
    module: str  # e.g., "client" or "envs.coding_env.client"
    class_name: str  # e.g., "CodingEnv"


@dataclass
class ActionMetadata:
    """Metadata about the action class."""
    module: str  # e.g., "client" or "envs.coding_env.client"
    class_name: str  # e.g., "CodeAction"


@dataclass
class ObservationMetadata:
    """Metadata about the observation class."""
    module: str  # e.g., "models" or "envs.coding_env.models"
    class_name: str  # e.g., "CodeObservation"


@dataclass
class EnvironmentManifest:
    """
    Parsed environment manifest containing all metadata.

    Attributes:
        name: Environment name (e.g., "echo_env")
        version: Version string (e.g., "0.1.0")
        description: Human-readable description
        client: Client class metadata
        action: Action class metadata
        observation: Observation class metadata
        spec_version: OpenEnv spec version (for openenv.yaml)
        runtime: Runtime type (e.g., "fastapi")
        app: App entry point (e.g., "server.app:app")
        port: Default port (e.g., 8000)
        raw_data: Raw dictionary from openenv.yaml (if parsed)
    """
    name: str
    version: str
    description: str
    client: ClientMetadata
    action: ActionMetadata
    observation: ObservationMetadata
    spec_version: Optional[int] = None
    runtime: Optional[str] = None
    app: Optional[str] = None
    port: Optional[int] = None
    raw_data: Optional[Dict[str, Any]] = None


def _infer_class_name_from_env_name(env_name: str, class_type: str) -> str:
    """
    Infer class name from environment directory name using conventions.

    Conventions:
    - Remove "_env" suffix: "echo_env" → "echo"
    - Convert to PascalCase: "browser_gym" → "BrowserGym"
    - Add class type suffix: "BrowserGym" + "Env" → "BrowserGymEnv"

    Special cases handled:
    - "browsergym" → "BrowserGymEnv", "BrowserGymAction" (capital G and Y)
    - "coding" → "CodingEnv", "CodeAction" (not CodingAction)
    - "dipg_safety" → "DIPGSafetyEnv", "DIPGAction" (all caps DIPG)
    - "finrl" → "FinRLEnv", "FinRLAction" (capital RL)
    - "openspiel" → "OpenSpielEnv", "OpenSpielAction" (capital S)
    - "sumo_rl" → "SumoRLEnv", "SumoAction" (capital RL for Env, just Sumo for Action)
    - "textarena" → "TextArenaEnv", "TextArenaAction" (capital A)

    Args:
        env_name: Environment directory name (e.g., "echo_env", "coding_env")
        class_type: Type of class ("client", "action", "observation")

    Returns:
        Inferred class name (e.g., "EchoEnv", "CodeAction")

    Examples:
        >>> _infer_class_name_from_env_name("echo_env", "client")
        'EchoEnv'
        >>> _infer_class_name_from_env_name("echo_env", "action")
        'EchoAction'
        >>> _infer_class_name_from_env_name("coding_env", "action")
        'CodeAction'
        >>> _infer_class_name_from_env_name("browsergym_env", "client")
        'BrowserGymEnv'
        >>> _infer_class_name_from_env_name("sumo_rl_env", "client")
        'SumoRLEnv'
        >>> _infer_class_name_from_env_name("dipg_safety_env", "client")
        'DIPGSafetyEnv'
    """
    # Remove "_env" suffix if present
    base_name = env_name[:-4] if env_name.endswith("_env") else env_name

    # Special case mapping for environments with non-standard capitalization
    # Format: base_name -> (EnvName, ActionName, ObservationName)
    special_cases = {
        "browsergym": ("BrowserGym", "BrowserGym", "BrowserGym"),
        "coding": ("Coding", "Code", "Code"),
        "dipg_safety": ("DIPGSafety", "DIPG", "DIPG"),
        "finrl": ("FinRL", "FinRL", "FinRL"),
        "openspiel": ("OpenSpiel", "OpenSpiel", "OpenSpiel"),
        "sumo_rl": ("SumoRL", "Sumo", "Sumo"),
        "textarena": ("TextArena", "TextArena", "TextArena"),
    }

    if base_name in special_cases:
        env_base, action_base, obs_base = special_cases[base_name]
        if class_type == "client":
            return f"{env_base}Env"
        elif class_type == "action":
            return f"{action_base}Action"
        elif class_type == "observation":
            return f"{obs_base}Observation"
        else:
            raise ValueError(f"Unknown class_type: {class_type}")
    else:
        # Standard PascalCase conversion
        # Split by underscore and capitalize each part
        parts = base_name.split("_")
        pascal_name = "".join(word.capitalize() for word in parts)

        # Apply class type suffix
        if class_type == "client":
            return f"{pascal_name}Env"
        elif class_type == "action":
            return f"{pascal_name}Action"
        elif class_type == "observation":
            return f"{pascal_name}Observation"
        else:
            raise ValueError(f"Unknown class_type: {class_type}")


def parse_manifest(manifest_path: Path) -> EnvironmentManifest:
    """
    Parse an openenv.yaml manifest file.

    Supports two formats:

    1. PR #160 format:
        spec_version: 1
        name: echo_env
        type: space
        runtime: fastapi
        app: server.app:app
        port: 8000

    2. Custom format (coding_env):
        name: coding_env
        version: "0.1.0"
        description: "Coding environment for OpenEnv"
        action: CodingAction
        observation: CodingObservation

    Args:
        manifest_path: Path to openenv.yaml file

    Returns:
        EnvironmentManifest with parsed data

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest is invalid or missing required fields
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        raise ValueError(f"Invalid manifest file: {manifest_path}")

    # Extract name (required in both formats)
    name = data.get("name")
    if not name:
        raise ValueError(f"Manifest missing 'name' field: {manifest_path}")

    # Extract version (optional, default to "0.1.0")
    version = data.get("version", "0.1.0")

    # Extract description (optional)
    description = data.get("description", f"{name} environment")

    # Extract spec_version (PR #160 format)
    spec_version = data.get("spec_version")

    # Extract runtime metadata (PR #160 format)
    runtime = data.get("runtime")
    app = data.get("app")
    port = data.get("port", 8000)

    # Determine client class
    if "client" in data and isinstance(data["client"], dict):
        # Explicit client metadata
        client = ClientMetadata(
            module=data["client"].get("module", "client"),
            class_name=data["client"].get("class", _infer_class_name_from_env_name(name, "client"))
        )
    else:
        # Infer from conventions
        client = ClientMetadata(
            module="client",
            class_name=_infer_class_name_from_env_name(name, "client")
        )

    # Determine action class
    if "action" in data:
        if isinstance(data["action"], dict):
            # Explicit action metadata
            action = ActionMetadata(
                module=data["action"].get("module", "client"),
                class_name=data["action"].get("class", _infer_class_name_from_env_name(name, "action"))
            )
        elif isinstance(data["action"], str):
            # Custom format: action: CodingAction
            action = ActionMetadata(
                module="client",
                class_name=data["action"]
            )
        else:
            raise ValueError(f"Invalid 'action' field in manifest: {manifest_path}")
    else:
        # Infer from conventions
        action = ActionMetadata(
            module="client",
            class_name=_infer_class_name_from_env_name(name, "action")
        )

    # Determine observation class
    if "observation" in data:
        if isinstance(data["observation"], dict):
            # Explicit observation metadata
            observation = ObservationMetadata(
                module=data["observation"].get("module", "models"),
                class_name=data["observation"].get("class", _infer_class_name_from_env_name(name, "observation"))
            )
        elif isinstance(data["observation"], str):
            # Custom format: observation: CodingObservation
            observation = ObservationMetadata(
                module="models",
                class_name=data["observation"]
            )
        else:
            raise ValueError(f"Invalid 'observation' field in manifest: {manifest_path}")
    else:
        # Infer from conventions
        observation = ObservationMetadata(
            module="models",
            class_name=_infer_class_name_from_env_name(name, "observation")
        )

    return EnvironmentManifest(
        name=name,
        version=version,
        description=description,
        client=client,
        action=action,
        observation=observation,
        spec_version=spec_version,
        runtime=runtime,
        app=app,
        port=port,
        raw_data=data
    )


def create_manifest_from_convention(env_dir: Path) -> EnvironmentManifest:
    """
    Create a manifest by inferring metadata from directory structure.

    This is used when no openenv.yaml exists. It uses naming conventions
    to infer the client, action, and observation class names.

    Args:
        env_dir: Path to environment directory (e.g., /path/to/echo_env)

    Returns:
        EnvironmentManifest with inferred data

    Examples:
        >>> manifest = create_manifest_from_convention(Path("src/envs/echo_env"))
        >>> manifest.name
        'echo_env'
        >>> manifest.client.class_name
        'EchoEnv'
        >>> manifest.action.class_name
        'EchoAction'
    """
    env_name = env_dir.name

    # Try to read version from pyproject.toml if it exists
    version = "0.1.0"
    pyproject_path = env_dir / "pyproject.toml"
    if pyproject_path.exists():
        try:
            import tomli
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomli.load(f)
                version = pyproject_data.get("project", {}).get("version", "0.1.0")
        except Exception:
            # If we can't parse pyproject.toml, use default
            pass

    return EnvironmentManifest(
        name=env_name,
        version=version,
        description=f"{env_name.replace('_', ' ').title()} environment",
        client=ClientMetadata(
            module="client",
            class_name=_infer_class_name_from_env_name(env_name, "client")
        ),
        action=ActionMetadata(
            module="client",
            class_name=_infer_class_name_from_env_name(env_name, "action")
        ),
        observation=ObservationMetadata(
            module="models",
            class_name=_infer_class_name_from_env_name(env_name, "observation")
        )
    )


def load_manifest(env_dir: Path) -> EnvironmentManifest:
    """
    Load environment manifest, trying openenv.yaml first, then falling back
    to convention-based inference.

    This is the main entry point for loading environment metadata.

    Args:
        env_dir: Path to environment directory

    Returns:
        EnvironmentManifest with environment metadata

    Examples:
        >>> # For echo_env (has openenv.yaml)
        >>> manifest = load_manifest(Path("src/envs/echo_env"))
        >>> manifest.name
        'echo_env'
        >>>
        >>> # For atari_env (no openenv.yaml, uses conventions)
        >>> manifest = load_manifest(Path("src/envs/atari_env"))
        >>> manifest.client.class_name
        'AtariEnv'
    """
    manifest_path = env_dir / "openenv.yaml"

    if manifest_path.exists():
        # Parse from openenv.yaml
        return parse_manifest(manifest_path)
    else:
        # Fall back to convention-based inference
        return create_manifest_from_convention(env_dir)
