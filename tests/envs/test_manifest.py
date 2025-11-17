# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Environment Manifest Parser
===========================================

Tests cover:
1. Convention-based class name inference
2. Parsing openenv.yaml (PR #160 format)
3. Parsing openenv.yaml (custom format)
4. Fallback to conventions
5. Error handling
"""

import pytest
import tempfile
from pathlib import Path
from textwrap import dedent

from envs._manifest import (
    _infer_class_name_from_env_name,
    parse_manifest,
    create_manifest_from_convention,
    load_manifest,
    EnvironmentManifest,
    ClientMetadata,
    ActionMetadata,
    ObservationMetadata,
)


class TestClassNameInference:
    """Test convention-based class name inference."""

    def test_infer_client_class_simple(self):
        """Test inferring client class name for simple environment."""
        assert _infer_class_name_from_env_name("echo_env", "client") == "EchoEnv"
        assert _infer_class_name_from_env_name("echo", "client") == "EchoEnv"

    def test_infer_action_class_simple(self):
        """Test inferring action class name for simple environment."""
        assert _infer_class_name_from_env_name("echo_env", "action") == "EchoAction"
        assert _infer_class_name_from_env_name("echo", "action") == "EchoAction"

    def test_infer_observation_class_simple(self):
        """Test inferring observation class name for simple environment."""
        assert _infer_class_name_from_env_name("echo_env", "observation") == "EchoObservation"

    def test_infer_with_underscores(self):
        """Test inferring class names with underscores (e.g., browser_gym)."""
        assert _infer_class_name_from_env_name("browsergym_env", "client") == "BrowsergymEnv"
        assert _infer_class_name_from_env_name("browsergym_env", "action") == "BrowsergymAction"

    def test_infer_special_case_coding(self):
        """Test special case: coding → CodeAction (not CodingAction)."""
        assert _infer_class_name_from_env_name("coding_env", "client") == "CodingEnv"
        assert _infer_class_name_from_env_name("coding_env", "action") == "CodeAction"
        assert _infer_class_name_from_env_name("coding_env", "observation") == "CodeObservation"

    def test_infer_special_case_sumo_rl(self):
        """Test special case: sumo_rl → SumoAction (not SumoRlAction)."""
        assert _infer_class_name_from_env_name("sumo_rl_env", "client") == "SumoRlEnv"
        assert _infer_class_name_from_env_name("sumo_rl_env", "action") == "SumoAction"

    def test_infer_atari(self):
        """Test Atari environment."""
        assert _infer_class_name_from_env_name("atari_env", "client") == "AtariEnv"
        assert _infer_class_name_from_env_name("atari_env", "action") == "AtariAction"

    def test_infer_connect4(self):
        """Test Connect4 environment (number in name)."""
        assert _infer_class_name_from_env_name("connect4_env", "client") == "Connect4Env"
        assert _infer_class_name_from_env_name("connect4_env", "action") == "Connect4Action"

    def test_infer_dipg_safety(self):
        """Test DIPG safety environment (multi-word)."""
        assert _infer_class_name_from_env_name("dipg_safety_env", "client") == "DipgSafetyEnv"
        assert _infer_class_name_from_env_name("dipg_safety_env", "action") == "DipgSafetyAction"

    def test_infer_invalid_class_type(self):
        """Test that invalid class type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown class_type"):
            _infer_class_name_from_env_name("echo_env", "invalid")


class TestParseManifest:
    """Test parsing openenv.yaml manifest files."""

    def test_parse_pr160_format(self, tmp_path):
        """Test parsing PR #160 standard format."""
        manifest_content = dedent("""
            spec_version: 1
            name: echo_env
            type: space
            runtime: fastapi
            app: server.app:app
            port: 8000
        """).strip()

        manifest_path = tmp_path / "openenv.yaml"
        manifest_path.write_text(manifest_content)

        manifest = parse_manifest(manifest_path)

        assert manifest.name == "echo_env"
        assert manifest.version == "0.1.0"  # Default
        assert manifest.spec_version == 1
        assert manifest.runtime == "fastapi"
        assert manifest.app == "server.app:app"
        assert manifest.port == 8000

        # Classes should be inferred
        assert manifest.client.class_name == "EchoEnv"
        assert manifest.client.module == "client"
        assert manifest.action.class_name == "EchoAction"
        assert manifest.action.module == "client"
        assert manifest.observation.class_name == "EchoObservation"
        assert manifest.observation.module == "models"

    def test_parse_custom_format_coding(self, tmp_path):
        """Test parsing custom format (coding_env style)."""
        manifest_content = dedent("""
            name: coding_env
            version: "0.1.0"
            description: "Coding environment for OpenEnv"
            action: CodeAction
            observation: CodeObservation
        """).strip()

        manifest_path = tmp_path / "openenv.yaml"
        manifest_path.write_text(manifest_content)

        manifest = parse_manifest(manifest_path)

        assert manifest.name == "coding_env"
        assert manifest.version == "0.1.0"
        assert manifest.description == "Coding environment for OpenEnv"

        # Client should be inferred
        assert manifest.client.class_name == "CodingEnv"
        assert manifest.client.module == "client"

        # Action and observation from manifest
        assert manifest.action.class_name == "CodeAction"
        assert manifest.action.module == "client"
        assert manifest.observation.class_name == "CodeObservation"
        assert manifest.observation.module == "models"

    def test_parse_extended_format(self, tmp_path):
        """Test parsing extended format with explicit class metadata."""
        manifest_content = dedent("""
            spec_version: 1
            name: custom_env
            version: "1.0.0"
            description: "Custom environment with explicit metadata"
            type: space
            runtime: fastapi
            app: server.app:app
            port: 8000

            client:
              module: custom_client
              class: MyCustomEnv

            action:
              module: custom_actions
              class: MyCustomAction

            observation:
              module: custom_models
              class: MyCustomObservation
        """).strip()

        manifest_path = tmp_path / "openenv.yaml"
        manifest_path.write_text(manifest_content)

        manifest = parse_manifest(manifest_path)

        assert manifest.name == "custom_env"
        assert manifest.version == "1.0.0"
        assert manifest.description == "Custom environment with explicit metadata"

        # Explicit metadata should be used
        assert manifest.client.class_name == "MyCustomEnv"
        assert manifest.client.module == "custom_client"
        assert manifest.action.class_name == "MyCustomAction"
        assert manifest.action.module == "custom_actions"
        assert manifest.observation.class_name == "MyCustomObservation"
        assert manifest.observation.module == "custom_models"

    def test_parse_missing_file(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        manifest_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            parse_manifest(manifest_path)

    def test_parse_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises ValueError."""
        manifest_path = tmp_path / "openenv.yaml"
        manifest_path.write_text("not: valid: yaml:")

        with pytest.raises(Exception):  # YAML parsing error
            parse_manifest(manifest_path)

    def test_parse_missing_name(self, tmp_path):
        """Test that missing 'name' field raises ValueError."""
        manifest_content = dedent("""
            spec_version: 1
            type: space
        """).strip()

        manifest_path = tmp_path / "openenv.yaml"
        manifest_path.write_text(manifest_content)

        with pytest.raises(ValueError, match="missing 'name' field"):
            parse_manifest(manifest_path)

    def test_parse_empty_file(self, tmp_path):
        """Test that empty file raises ValueError."""
        manifest_path = tmp_path / "openenv.yaml"
        manifest_path.write_text("")

        with pytest.raises(ValueError, match="Invalid manifest"):
            parse_manifest(manifest_path)


class TestCreateManifestFromConvention:
    """Test creating manifest from directory conventions."""

    def test_create_from_simple_env(self, tmp_path):
        """Test creating manifest for simple environment."""
        env_dir = tmp_path / "echo_env"
        env_dir.mkdir()

        manifest = create_manifest_from_convention(env_dir)

        assert manifest.name == "echo_env"
        assert manifest.version == "0.1.0"
        assert manifest.description == "Echo Env environment"
        assert manifest.client.class_name == "EchoEnv"
        assert manifest.action.class_name == "EchoAction"
        assert manifest.observation.class_name == "EchoObservation"

    def test_create_from_complex_env(self, tmp_path):
        """Test creating manifest for complex environment name."""
        env_dir = tmp_path / "browsergym_env"
        env_dir.mkdir()

        manifest = create_manifest_from_convention(env_dir)

        assert manifest.name == "browsergym_env"
        assert manifest.client.class_name == "BrowsergymEnv"
        assert manifest.action.class_name == "BrowsergymAction"

    def test_create_from_coding_env(self, tmp_path):
        """Test creating manifest for coding_env (special case)."""
        env_dir = tmp_path / "coding_env"
        env_dir.mkdir()

        manifest = create_manifest_from_convention(env_dir)

        assert manifest.name == "coding_env"
        assert manifest.client.class_name == "CodingEnv"
        assert manifest.action.class_name == "CodeAction"
        assert manifest.observation.class_name == "CodeObservation"

    def test_create_reads_version_from_pyproject(self, tmp_path):
        """Test that version is read from pyproject.toml if available."""
        env_dir = tmp_path / "test_env"
        env_dir.mkdir()

        # Create pyproject.toml with version
        pyproject_content = dedent("""
            [project]
            name = "test-env"
            version = "2.5.3"
        """).strip()
        (env_dir / "pyproject.toml").write_text(pyproject_content)

        manifest = create_manifest_from_convention(env_dir)

        assert manifest.version == "2.5.3"


class TestLoadManifest:
    """Test load_manifest function (main entry point)."""

    def test_load_with_yaml(self, tmp_path):
        """Test loading when openenv.yaml exists."""
        env_dir = tmp_path / "echo_env"
        env_dir.mkdir()

        manifest_content = dedent("""
            spec_version: 1
            name: echo_env
            version: "1.2.3"
            type: space
            runtime: fastapi
            app: server.app:app
            port: 8000
        """).strip()

        (env_dir / "openenv.yaml").write_text(manifest_content)

        manifest = load_manifest(env_dir)

        # Should load from YAML
        assert manifest.name == "echo_env"
        assert manifest.version == "1.2.3"
        assert manifest.spec_version == 1

    def test_load_without_yaml(self, tmp_path):
        """Test loading when openenv.yaml doesn't exist (fallback to conventions)."""
        env_dir = tmp_path / "atari_env"
        env_dir.mkdir()

        manifest = load_manifest(env_dir)

        # Should fall back to conventions
        assert manifest.name == "atari_env"
        assert manifest.version == "0.1.0"
        assert manifest.client.class_name == "AtariEnv"
        assert manifest.action.class_name == "AtariAction"
        assert manifest.spec_version is None  # Not from YAML

    def test_load_with_pyproject_only(self, tmp_path):
        """Test loading with pyproject.toml but no openenv.yaml."""
        env_dir = tmp_path / "test_env"
        env_dir.mkdir()

        pyproject_content = dedent("""
            [project]
            name = "test-env"
            version = "3.0.0"
        """).strip()
        (env_dir / "pyproject.toml").write_text(pyproject_content)

        manifest = load_manifest(env_dir)

        # Should use version from pyproject.toml
        assert manifest.name == "test_env"
        assert manifest.version == "3.0.0"
        assert manifest.client.class_name == "TestEnv"


class TestManifestDataclasses:
    """Test manifest dataclass creation and properties."""

    def test_client_metadata_creation(self):
        """Test creating ClientMetadata."""
        client = ClientMetadata(module="client", class_name="EchoEnv")
        assert client.module == "client"
        assert client.class_name == "EchoEnv"

    def test_action_metadata_creation(self):
        """Test creating ActionMetadata."""
        action = ActionMetadata(module="client", class_name="EchoAction")
        assert action.module == "client"
        assert action.class_name == "EchoAction"

    def test_observation_metadata_creation(self):
        """Test creating ObservationMetadata."""
        obs = ObservationMetadata(module="models", class_name="EchoObservation")
        assert obs.module == "models"
        assert obs.class_name == "EchoObservation"

    def test_environment_manifest_creation(self):
        """Test creating full EnvironmentManifest."""
        manifest = EnvironmentManifest(
            name="echo_env",
            version="0.1.0",
            description="Test environment",
            client=ClientMetadata(module="client", class_name="EchoEnv"),
            action=ActionMetadata(module="client", class_name="EchoAction"),
            observation=ObservationMetadata(module="models", class_name="EchoObservation"),
            spec_version=1,
            runtime="fastapi",
            app="server.app:app",
            port=8000
        )

        assert manifest.name == "echo_env"
        assert manifest.version == "0.1.0"
        assert manifest.client.class_name == "EchoEnv"
        assert manifest.action.class_name == "EchoAction"
        assert manifest.observation.class_name == "EchoObservation"
        assert manifest.spec_version == 1
        assert manifest.port == 8000
