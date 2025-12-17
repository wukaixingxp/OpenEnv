"""
envs/julia_env/models.py
--------------------------------
Action/Observation types for the Julia environment.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server import Action, Observation, State


class JuliaAction(Action):
    """
    Represents a single Julia code execution request with tests.

    Attributes:
        core_code: Main Julia code to execute (e.g., function definition)
        test_code: Test code to execute (e.g., unit tests)
    """
    core_code: str
    test_code: str = ""


class JuliaObservation(Observation):
    """
    Result of executing Julia code in the environment.

    Attributes:
        stdout: Standard output from code execution
        stderr: Standard error from code execution
        exit_code: Exit code (0 = success, non-zero = error)
        tests_passed: Number of tests that passed
        tests_failed: Number of tests that failed
        code_compiles: Whether the code compiled/executed without syntax errors
        reward: Calculated reward based on test results
    """
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    code_compiles: bool = True
    reward: float = 0.0


class JuliaState(State):
    """State for Julia environment with persistent execution context."""
    last_exit_code: int = 0
    last_code_compiles: bool = True
    total_tests_passed: int = 0
    total_tests_failed: int = 0
