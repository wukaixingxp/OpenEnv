# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment Registry for AutoEnv and AutoAction
================================================

This module provides a centralized registry mapping environment names to
their corresponding client classes, action classes, and default Docker
image names.

The registry enables the AutoEnv and AutoAction classes to automatically
instantiate the correct environment and action types based on Docker
image names.
"""

from typing import Any, Dict

# Registry structure:
# env_key: (module_path, env_class_name, action_class_name,
#           default_image, special_notes)
ENV_REGISTRY: Dict[str, Dict[str, Any]] = {
    "atari": {
        "module": "envs.atari_env",
        "env_class": "AtariEnv",
        "action_class": "AtariAction",
        "default_image": "atari-env:latest",
        "description": "Atari 2600 games environment (100+ games)",
        "special_requirements": None,
        "supported_features": [
            "Multiple games (100+)",
            "RGB/grayscale/RAM observations",
            "Configurable action spaces (minimal/full)",
            "Frame skipping and sticky actions",
        ],
    },
    "browsergym": {
        "module": "envs.browsergym_env",
        "env_class": "BrowserGymEnv",
        "action_class": "BrowserGymAction",
        "default_image": "browsergym-env:latest",
        "description": "Web browsing environment with multiple benchmarks",
        "special_requirements": "WebArena tasks require backend setup with env vars",
        "supported_features": [
            "MiniWoB/WebArena/VisualWebArena benchmarks",
            "Natural language actions",
            "Multi-modal observations (text/visual)",
        ],
    },
    "chat": {
        "module": "envs.chat_env",
        "env_class": "ChatEnv",
        "action_class": "ChatAction",
        "default_image": "chat-env:latest",
        "description": "Chat environment with tokenization support",
        "special_requirements": None,
        "supported_features": [
            "PyTorch tensor handling",
            "Hugging Face chat format",
            "Optional tokenization with TOKENIZER_NAME env var",
        ],
    },
    "coding": {
        "module": "envs.coding_env",
        "env_class": "CodingEnv",
        "action_class": "CodeAction",
        "default_image": "coding-env:latest",
        "description": "Python code execution environment",
        "special_requirements": None,
        "supported_features": [
            "Python code execution",
            "Persistent execution context",
            "stdout/stderr/exit_code capture",
        ],
    },
    "connect4": {
        "module": "envs.connect4_env",
        "env_class": "Connect4Env",
        "action_class": "Connect4Action",
        "default_image": "connect4-env:latest",
        "description": "Connect Four board game environment",
        "special_requirements": None,
        "supported_features": [
            "Two-player game (6x7 grid)",
            "Legal actions masking",
            "Turn tracking",
        ],
    },
    "dipg": {
        "module": "envs.dipg_safety_env",
        "env_class": "DIPGSafetyEnv",
        "action_class": "DIPGAction",
        "default_image": "dipg-env:latest",
        "description": "DIPG safety-critical medical decision environment",
        "special_requirements": "Requires DIPG_DATASET_PATH env var pointing to dataset",
        "supported_features": [
            "Safety-critical medical domain",
            "LLM response scoring",
            "Conflict/abstention rewards",
        ],
    },
    "echo": {
        "module": "envs.echo_env",
        "env_class": "EchoEnv",
        "action_class": "EchoAction",
        "default_image": "echo-env:latest",
        "description": "Simple echo test environment",
        "special_requirements": None,
        "supported_features": [
            "Message echoing",
            "Basic HTTP server testing",
        ],
    },
    "finrl": {
        "module": "envs.finrl_env",
        "env_class": "FinRLEnv",
        "action_class": "FinRLAction",
        "default_image": "finrl-env:latest",
        "description": "Financial trading environment",
        "special_requirements": "Optional FINRL_CONFIG_PATH env var for custom configuration",
        "supported_features": [
            "Stock trading simulation",
            "Technical indicators",
            "Custom configuration support",
        ],
    },
    "git": {
        "module": "envs.git_env",
        "env_class": "GitEnv",
        "action_class": "GitAction",
        "default_image": "git-env:latest",
        "description": "Git repository management with Gitea integration",
        "special_requirements": None,
        "supported_features": [
            "Repository cloning",
            "Git command execution",
            "Gitea server integration",
        ],
    },
    "openspiel": {
        "module": "envs.openspiel_env",
        "env_class": "OpenSpielEnv",
        "action_class": "OpenSpielAction",
        "default_image": "openspiel-env:latest",
        "description": "OpenSpiel game environment (multiple games)",
        "special_requirements": None,
        "supported_features": [
            "6 supported games (catch/tic-tac-toe/kuhn_poker/cliff_walking/2048/blackjack)",
            "Single and multi-player support",
            "Optional opponent policies",
        ],
    },
    "sumo_rl": {
        "module": "envs.sumo_rl_env",
        "env_class": "SumoRLEnv",
        "action_class": "SumoAction",
        "default_image": "sumo-rl-env:latest",
        "description": "SUMO traffic signal control environment",
        "special_requirements": "Custom network files can be provided via volume mounts",
        "supported_features": [
            "Traffic signal control",
            "SUMO simulator integration",
            "Multiple reward functions",
            "Phase-based actions with configurable timings",
        ],
    },
    "textarena": {
        "module": "envs.textarena_env",
        "env_class": "TextArenaEnv",
        "action_class": "TextArenaAction",
        "default_image": "textarena-env:latest",
        "description": "Text-based game environment (word games, reasoning tasks)",
        "special_requirements": None,
        "supported_features": [
            "Word and reasoning games",
            "Multi-agent support",
            "Environment configuration via kwargs",
        ],
    },
    "julia": {
        "module": "envs.julia_env",
        "env_class": "JuliaEnv",
        "action_class": "JuliaAction",
        "default_image": "julia-env:latest",
        "description": "Julia code execution environment with test support",
        "special_requirements": None,
        "supported_features": [
            "Julia code execution",
            "Test execution with @test macros",
            "stdout/stderr/exit_code capture",
            "Code compilation checking",
        ],
    },
}

# Deprecated or removed environments
DEPRECATED_ENVS: Dict[str, str] = {}


def get_env_info(env_key: str) -> Dict[str, Any]:
    """
    Get environment information from registry.

    Args:
        env_key: Environment key (e.g., "coding", "atari")

    Returns:
        Dictionary with environment information

    Raises:
        ValueError: If environment key is not found in registry
    """
    env_key = env_key.lower()

    # Check if deprecated
    if env_key in DEPRECATED_ENVS:
        raise ValueError(DEPRECATED_ENVS[env_key])

    # Get from registry
    if env_key not in ENV_REGISTRY:
        # Try to suggest similar environment names
        from difflib import get_close_matches

        suggestions = get_close_matches(env_key, ENV_REGISTRY.keys(), n=3, cutoff=0.6)
        suggestion_str = ""
        if suggestions:
            suggestion_str = f" Did you mean: {', '.join(suggestions)}?"

        raise ValueError(
            f"Unknown environment '{env_key}'. "
            f"Supported environments: {', '.join(sorted(ENV_REGISTRY.keys()))}.{suggestion_str}"
        )

    return ENV_REGISTRY[env_key]


def list_available_environments() -> Dict[str, str]:
    """
    List all available environments with their descriptions.

    Returns:
        Dictionary mapping environment keys to descriptions
    """
    return {key: info["description"] for key, info in ENV_REGISTRY.items()}


def get_all_env_keys() -> list[str]:
    """Get list of all registered environment keys."""
    return sorted(ENV_REGISTRY.keys())
