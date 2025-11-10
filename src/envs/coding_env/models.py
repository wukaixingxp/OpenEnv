"""
envs/coding_env/models.py
--------------------------------
Action/Observation types for the Coding environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.env_server import Action, Observation, State


@dataclass(kw_only=True)
class CodeAction(Action):
    """
    Represents a single code execution request with optional tests.

    Attributes:
        code: Main code to execute
        test_code: Optional test code to execute (e.g., assertions/unit tests)
    """
    code: str
    test_code: str = ""


@dataclass
class CodeObservation(Observation):
    """
    Result of executing code in the environment.

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


@dataclass
class CodeState(State):
    """State for CodeAct environment with persistent execution context."""
    last_exit_code: int = 0
    last_code_compiles: bool = True
    total_tests_passed: int = 0
    total_tests_failed: int = 0
