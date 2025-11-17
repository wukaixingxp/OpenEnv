# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for Environment Auto-Discovery System
=================================================

Tests cover:
1. Environment discovery from directories
2. Cache loading and saving
3. Validation of environment directories
4. Getting specific environments
5. Listing environments
6. Error handling
"""

import pytest
import json
from pathlib import Path
from textwrap import dedent

from envs._discovery import (
    EnvironmentDiscovery,
    EnvironmentInfo,
    get_discovery,
    reset_discovery,
)


class TestEnvironmentInfo:
    """Test EnvironmentInfo dataclass and methods."""

    def test_environment_info_creation(self):
        """Test creating EnvironmentInfo instance."""
        env_info = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            version="0.1.0",
            description="Echo environment",
            env_dir="/path/to/echo_env",
            client_module_path="envs.echo_env.client",
            action_module_path="envs.echo_env.client",
            observation_module_path="envs.echo_env.models",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest"
        )

        assert env_info.env_key == "echo"
        assert env_info.name == "echo_env"
        assert env_info.client_class_name == "EchoEnv"
        assert env_info.default_image == "echo-env:latest"


class TestEnvironmentDiscoveryValidation:
    """Test environment directory validation."""

    def test_is_valid_env_dir_with_client(self, tmp_path):
        """Test validation with client.py present."""
        env_dir = tmp_path / "test_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# client code")

        discovery = EnvironmentDiscovery(tmp_path)
        assert discovery._is_valid_env_dir(env_dir)

    def test_is_valid_env_dir_with_server(self, tmp_path):
        """Test validation with server/ directory present."""
        env_dir = tmp_path / "test_env"
        env_dir.mkdir()
        (env_dir / "server").mkdir()

        discovery = EnvironmentDiscovery(tmp_path)
        assert discovery._is_valid_env_dir(env_dir)

    def test_is_valid_env_dir_with_both(self, tmp_path):
        """Test validation with both client.py and server/ present."""
        env_dir = tmp_path / "test_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# client")
        (env_dir / "server").mkdir()

        discovery = EnvironmentDiscovery(tmp_path)
        assert discovery._is_valid_env_dir(env_dir)

    def test_is_valid_env_dir_empty(self, tmp_path):
        """Test validation with empty directory (should be invalid)."""
        env_dir = tmp_path / "empty_env"
        env_dir.mkdir()

        discovery = EnvironmentDiscovery(tmp_path)
        assert not discovery._is_valid_env_dir(env_dir)

    def test_is_valid_env_dir_hidden(self, tmp_path):
        """Test that hidden directories are skipped."""
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "client.py").write_text("# client")

        discovery = EnvironmentDiscovery(tmp_path)
        assert not discovery._is_valid_env_dir(hidden_dir)

    def test_is_valid_env_dir_underscore(self, tmp_path):
        """Test that underscore-prefixed directories are skipped."""
        under_dir = tmp_path / "_private"
        under_dir.mkdir()
        (under_dir / "client.py").write_text("# client")

        discovery = EnvironmentDiscovery(tmp_path)
        assert not discovery._is_valid_env_dir(under_dir)

    def test_is_valid_env_dir_file(self, tmp_path):
        """Test that files are not valid (only directories)."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# code")

        discovery = EnvironmentDiscovery(tmp_path)
        assert not discovery._is_valid_env_dir(test_file)


class TestEnvironmentDiscovery:
    """Test main discovery functionality."""

    def test_discover_simple_environment(self, tmp_path):
        """Test discovering a simple environment."""
        # Create echo_env
        env_dir = tmp_path / "echo_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# echo client")

        discovery = EnvironmentDiscovery(tmp_path)
        environments = discovery.discover(use_cache=False)

        assert "echo" in environments
        env = environments["echo"]
        assert env.name == "echo_env"
        assert env.client_class_name == "EchoEnv"
        assert env.action_class_name == "EchoAction"
        assert env.observation_class_name == "EchoObservation"

    def test_discover_multiple_environments(self, tmp_path):
        """Test discovering multiple environments."""
        # Create multiple environments
        for env_name in ["echo_env", "coding_env", "atari_env"]:
            env_dir = tmp_path / env_name
            env_dir.mkdir()
            (env_dir / "client.py").write_text("# client")

        discovery = EnvironmentDiscovery(tmp_path)
        environments = discovery.discover(use_cache=False)

        assert len(environments) == 3
        assert "echo" in environments
        assert "coding" in environments
        assert "atari" in environments

    def test_discover_with_openenv_yaml(self, tmp_path):
        """Test discovering environment with openenv.yaml."""
        env_dir = tmp_path / "test_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# client")

        # Create openenv.yaml
        manifest_content = dedent("""
            spec_version: 1
            name: test_env
            version: "2.0.0"
            description: "Test environment with manifest"
            type: space
            runtime: fastapi
            app: server.app:app
            port: 8000
        """).strip()
        (env_dir / "openenv.yaml").write_text(manifest_content)

        discovery = EnvironmentDiscovery(tmp_path)
        environments = discovery.discover(use_cache=False)

        assert "test" in environments
        env = environments["test"]
        assert env.version == "2.0.0"
        assert env.description == "Test environment with manifest"
        assert env.spec_version == 1

    def test_discover_skips_invalid_dirs(self, tmp_path):
        """Test that discovery skips invalid directories."""
        # Create valid environment
        valid_env = tmp_path / "valid_env"
        valid_env.mkdir()
        (valid_env / "client.py").write_text("# client")

        # Create invalid directories
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "_private").mkdir()
        (tmp_path / "empty_dir").mkdir()

        discovery = EnvironmentDiscovery(tmp_path)
        environments = discovery.discover(use_cache=False)

        # Only valid_env should be discovered
        assert len(environments) == 1
        assert "valid" in environments

    def test_discover_handles_broken_manifest(self, tmp_path):
        """Test that discovery handles broken manifest gracefully."""
        # Create environment with broken manifest
        env_dir = tmp_path / "broken_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# client")
        (env_dir / "openenv.yaml").write_text("invalid: yaml: format:")

        # Create valid environment
        valid_env = tmp_path / "valid_env"
        valid_env.mkdir()
        (valid_env / "client.py").write_text("# client")

        discovery = EnvironmentDiscovery(tmp_path)
        environments = discovery.discover(use_cache=False)

        # Should discover valid_env but skip broken_env
        assert "valid" in environments
        assert "broken" not in environments

    def test_get_environment(self, tmp_path):
        """Test getting a specific environment."""
        env_dir = tmp_path / "echo_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# client")

        discovery = EnvironmentDiscovery(tmp_path)
        env = discovery.get_environment("echo")

        assert env is not None
        assert env.name == "echo_env"
        assert env.client_class_name == "EchoEnv"

    def test_get_nonexistent_environment(self, tmp_path):
        """Test getting a non-existent environment."""
        discovery = EnvironmentDiscovery(tmp_path)
        env = discovery.get_environment("nonexistent")

        assert env is None

    def test_discover_nonexistent_directory(self, tmp_path):
        """Test discovery with non-existent directory."""
        nonexistent = tmp_path / "nonexistent"

        discovery = EnvironmentDiscovery(nonexistent)
        environments = discovery.discover(use_cache=False)

        assert len(environments) == 0


class TestDiscoveryCache:
    """Test caching functionality."""

    def test_save_and_load_cache(self, tmp_path):
        """Test saving and loading discovery cache."""
        # Create environment
        env_dir = tmp_path / "echo_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# client")

        # First discovery (creates cache)
        discovery = EnvironmentDiscovery(tmp_path)
        envs1 = discovery.discover(use_cache=False)

        # Check cache file was created
        cache_file = tmp_path / ".discovery_cache.json"
        assert cache_file.exists()

        # Second discovery (loads from cache)
        discovery2 = EnvironmentDiscovery(tmp_path)
        envs2 = discovery2.discover(use_cache=True)

        # Should have same results
        assert envs1.keys() == envs2.keys()
        assert envs2["echo"].name == "echo_env"

    def test_cache_invalidation(self, tmp_path):
        """Test that cache can be cleared."""
        env_dir = tmp_path / "echo_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# client")

        discovery = EnvironmentDiscovery(tmp_path)
        discovery.discover(use_cache=False)

        # Clear cache
        discovery.clear_cache()

        # Cache file should be removed
        cache_file = tmp_path / ".discovery_cache.json"
        assert not cache_file.exists()

    def test_discover_without_cache(self, tmp_path):
        """Test discovery without using cache."""
        env_dir = tmp_path / "echo_env"
        env_dir.mkdir()
        (env_dir / "client.py").write_text("# client")

        discovery = EnvironmentDiscovery(tmp_path)

        # First discovery with use_cache=False
        envs1 = discovery.discover(use_cache=False)

        # Add new environment
        env_dir2 = tmp_path / "coding_env"
        env_dir2.mkdir()
        (env_dir2 / "client.py").write_text("# client")

        # Second discovery with use_cache=False should find new environment
        envs2 = discovery.discover(use_cache=False)

        assert len(envs2) == 2
        assert "echo" in envs2
        assert "coding" in envs2


class TestGlobalDiscovery:
    """Test global discovery instance."""

    def test_get_discovery_default(self):
        """Test getting global discovery instance."""
        reset_discovery()  # Start fresh
        discovery = get_discovery()

        assert discovery is not None
        assert isinstance(discovery, EnvironmentDiscovery)

    def test_get_discovery_custom_dir(self, tmp_path):
        """Test getting global discovery with custom directory."""
        reset_discovery()  # Start fresh
        discovery = get_discovery(envs_dir=tmp_path)

        assert discovery.envs_dir == tmp_path

    def test_get_discovery_singleton(self):
        """Test that get_discovery returns same instance."""
        reset_discovery()  # Start fresh
        discovery1 = get_discovery()
        discovery2 = get_discovery()

        assert discovery1 is discovery2

    def test_reset_discovery(self):
        """Test resetting global discovery instance."""
        discovery1 = get_discovery()
        reset_discovery()
        discovery2 = get_discovery()

        # Should be different instances after reset
        assert discovery1 is not discovery2


class TestListEnvironments:
    """Test list_environments output."""

    def test_list_environments(self, tmp_path, capsys):
        """Test listing environments."""
        # Create multiple environments
        for env_name in ["echo_env", "coding_env"]:
            env_dir = tmp_path / env_name
            env_dir.mkdir()
            (env_dir / "client.py").write_text("# client")

        discovery = EnvironmentDiscovery(tmp_path)
        discovery.list_environments()

        # Check output
        captured = capsys.readouterr()
        assert "Discovered Environments:" in captured.out
        assert "echo" in captured.out
        assert "coding" in captured.out
        assert "Total: 2 environments" in captured.out

    def test_list_empty(self, tmp_path, capsys):
        """Test listing when no environments found."""
        discovery = EnvironmentDiscovery(tmp_path)
        discovery.list_environments()

        captured = capsys.readouterr()
        assert "Total: 0 environments" in captured.out


class TestCreateEnvInfo:
    """Test _create_env_info method."""

    def test_create_env_info_simple(self, tmp_path):
        """Test creating EnvironmentInfo from manifest."""
        from envs._manifest import create_manifest_from_convention

        env_dir = tmp_path / "echo_env"
        env_dir.mkdir()

        manifest = create_manifest_from_convention(env_dir)
        discovery = EnvironmentDiscovery(tmp_path)
        env_info = discovery._create_env_info(manifest, env_dir)

        assert env_info.env_key == "echo"
        assert env_info.name == "echo_env"
        assert env_info.default_image == "echo-env:latest"
        assert env_info.client_module_path == "envs.echo_env.client"

    def test_create_env_info_with_underscores(self, tmp_path):
        """Test creating EnvironmentInfo with underscores in name."""
        from envs._manifest import create_manifest_from_convention

        env_dir = tmp_path / "sumo_rl_env"
        env_dir.mkdir()

        manifest = create_manifest_from_convention(env_dir)
        discovery = EnvironmentDiscovery(tmp_path)
        env_info = discovery._create_env_info(manifest, env_dir)

        assert env_info.env_key == "sumo_rl"
        assert env_info.default_image == "sumo-rl-env:latest"
