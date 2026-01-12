"""
envs/coding_env/models.py
--------------------------------
Action/Observation types for the Coding environment.
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import Action, Observation, State


class CodeAction(Action):
    """
    Represents a single code execution request.

    Attributes:
        code: Core Python code to execute
        test_code: Optional test code to execute. If provided, runs after core code.
    """

    code: str
    test_code: str | None = None


class CodeObservation(Observation):
    """
    Result of executing code in the environment.

    Attributes:
        stdout: Standard output from Python execution
        stderr: Standard error from Python execution
        exit_code: Exit code (0 = success, non-zero = error)
        tests_passed: Number of tests passed (if tests were run)
        tests_failed: Number of tests failed (if tests were run)
        code_compiles: Whether the core code compiled/executed successfully
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    code_compiles: bool = True


class CodeState(State):
    """
    State for CodeAct environment with persistent execution context.

    Attributes:
        last_exit_code: Exit code from last execution
        last_code_compiles: Whether the last code compiled successfully
        total_tests_passed: Cumulative tests passed in episode
        total_tests_failed: Cumulative tests failed in episode
    """

    last_exit_code: int = 0
    last_code_compiles: bool = True
    total_tests_passed: int = 0
    total_tests_failed: int = 0
