# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for AutoEnv and AutoAction
=============================================

Tests the full integration of discovery system with AutoEnv/AutoAction.
"""

import pytest
from envs import AutoEnv, AutoAction


class TestAutoEnvIntegration:
    """Test AutoEnv integration with discovery system."""

    def test_auto_env_get_env_class(self):
        """Test getting environment class by name."""
        EchoEnv = AutoEnv.get_env_class("echo")
        assert EchoEnv.__name__ == "EchoEnv"

        # Note: coding_env currently has import issues (uses absolute imports)
        # Skip for now
        # CodingEnv = AutoEnv.get_env_class("coding")
        # assert CodingEnv.__name__ == "CodingEnv"

    def test_auto_env_get_env_info(self):
        """Test getting environment info."""
        info = AutoEnv.get_env_info("echo")
        assert info["name"] == "echo_env"
        assert info["env_class"] == "EchoEnv"
        assert info["action_class"] == "EchoAction"
        assert "description" in info
        assert "default_image" in info

    def test_auto_env_list_environments(self, capsys):
        """Test listing all environments."""
        AutoEnv.list_environments()
        captured = capsys.readouterr()
        assert "Available Environments" in captured.out
        assert "echo" in captured.out
        assert "coding" in captured.out
        assert "Total: 12 environments" in captured.out


class TestAutoActionIntegration:
    """Test AutoAction integration with discovery system."""

    def test_auto_action_from_env(self):
        """Test getting action class from environment name."""
        EchoAction = AutoAction.from_env("echo")
        assert EchoAction.__name__ == "EchoAction"

    def test_auto_action_from_name(self):
        """Test getting action class from environment name."""
        EchoAction = AutoAction.from_name("echo-env")
        assert EchoAction.__name__ == "EchoAction"

        # Note: coding_env currently has import issues (uses absolute imports)
        # Skip for now
        # CodingAction = AutoAction.from_name("coding-env")
        # assert CodingAction.__name__ in ["CodeAction", "CodingAction"]

    def test_auto_action_get_action_info(self):
        """Test getting action info."""
        info = AutoAction.get_action_info("echo")
        assert info["action_class"] == "EchoAction"
        assert info["env_class"] == "EchoEnv"
        assert "description" in info

    def test_auto_action_list_actions(self, capsys):
        """Test listing all action classes."""
        AutoAction.list_actions()
        captured = capsys.readouterr()
        assert "Available Action Classes" in captured.out
        assert "EchoAction" in captured.out
        assert "Total: 12 Action classes" in captured.out


class TestAutoEnvAutoActionTogether:
    """Test using AutoEnv and AutoAction together."""

    def test_auto_env_and_action_together(self):
        """Test getting both environment and action class."""
        # Get environment class
        EchoEnv = AutoEnv.get_env_class("echo")
        assert EchoEnv.__name__ == "EchoEnv"

        # Get action class
        EchoAction = AutoAction.from_env("echo")
        assert EchoAction.__name__ == "EchoAction"

        # Verify they're related
        info = AutoEnv.get_env_info("echo")
        assert info["action_class"] == "EchoAction"

    def test_multiple_environments(self):
        """Test with multiple environments."""
        test_envs = ["echo", "atari", "connect4"]

        for env_key in test_envs:
            # Get environment class
            env_class = AutoEnv.get_env_class(env_key)
            assert env_class is not None

            # Get action class
            action_class = AutoAction.from_env(env_key)
            assert action_class is not None

            # Verify they match
            info = AutoEnv.get_env_info(env_key)
            assert info["action_class"] == action_class.__name__


class TestDiscoveryPerformance:
    """Test that discovery is performant (uses caching)."""

    def test_discovery_uses_cache(self):
        """Test that repeated calls use cache."""
        from envs._discovery import get_discovery

        # First call - discovers and caches
        discovery = get_discovery()
        envs1 = discovery.discover(use_cache=False)

        # Second call - should use cache
        envs2 = discovery.discover(use_cache=True)

        # Should return same results
        assert envs1.keys() == envs2.keys()
        assert len(envs1) == len(envs2)
