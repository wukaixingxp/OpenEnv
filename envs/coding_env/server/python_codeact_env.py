# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python Code Action Environment.

This module provides a server-side environment implementation for executing
Python code actions using PyExecutor.
"""

import re
import uuid

from openenv.core.env_server.interfaces import Action, Environment, Observation
from .python_executor import PyExecutor

from ..models import CodeAction, CodeObservation, CodeState
from .transforms import create_safe_coding_transform


class PythonCodeActEnv(Environment):
    """
    Python Code Action Environment for executing code and tracking state.

    This environment executes Python code submitted as CodeAction during step,
    maintains the last exit code in its state, and returns results wrapped
    in CodeObservation.

    Args:
        transform: Optional transform to apply to observations
        additional_imports: List of additional module imports to authorize
                          (e.g., ["numpy", "pandas", "matplotlib"])

    Example:
        >>> env = PythonCodeActEnv()
        >>> obs = env.reset()
        >>> action = CodeAction(code="print('Hello, World!')")
        >>> obs = env.step(action)
        >>> print(obs.stdout)  # "Hello, World!\n"
        >>> print(obs.exit_code)  # 0
        >>> print(env.state.last_exit_code)  # 0

    Example with test_code:
        >>> env = PythonCodeActEnv()
        >>> obs = env.reset()
        >>> action = CodeAction(
        ...     code='def add(a, b): return a + b',
        ...     test_code='assert add(2, 3) == 5'
        ... )
        >>> obs = env.step(action)
        >>> print(obs.tests_passed)  # 1
        >>> print(obs.code_compiles)  # True
    """

    def __init__(
        self,
    ):
        self.transform = create_safe_coding_transform()
        self._executor = PyExecutor()
        self._state = CodeState()

    def reset(self, **kwargs) -> Observation:
        """
        Reset environment and start fresh execution session.

        Returns:
            Initial observation with empty stdout/stderr and exit_code=0
        """
        # Initialize fresh state
        self._state = CodeState(episode_id=str(uuid.uuid4()), step_count=0)
        self._state.last_exit_code = 0
        self._state.last_code_compiles = True

        # Reset executor to clear any previously defined variables/functions
        self._executor = PyExecutor()

        # Reset transform to clear any accumulated state
        self.transform = create_safe_coding_transform()

        # Return initial observation
        observation = CodeObservation(
            stdout="",
            stderr="",
            exit_code=0,
            reward=0.0,
            metadata={"code": "", "test_code": ""},
            tests_passed=0,
            tests_failed=0,
            code_compiles=True,
        )

        return self._apply_transform(observation)

    def step(self, action: Action, **kwargs) -> Observation:
        """
        Execute code action and return observation.

        Optimized single-pass execution:
        - Runs code + test_code together
        - Infers compilation status from combined execution
        - 2x faster than double execution

        Args:
            action: CodeAction containing the code to execute
            **kwargs: Optional parameters including:
                - timeout: Execution timeout in seconds (default: 120)

        Returns:
            CodeObservation with execution results (stdout, stderr, exit_code)

        Raises:
            ValueError: If action is not a CodeAction instance
        """
        if not isinstance(action, CodeAction):
            raise ValueError(f"Expected CodeAction, got {type(action)}")

        # Single execution: Run code + test_code together (if test_code provided)
        if action.test_code:
            combined_code = action.code + "\n\n" + action.test_code
        else:
            combined_code = action.code

        # Execute the code using PyExecutor
        result = self._executor.run(combined_code)

        # Parse test results from execution output
        tests_passed, tests_failed = self._parse_test_results(
            result.stdout, result.stderr
        )

        # Infer compilation status from execution
        # If tests ran, code compiled successfully
        # If exit_code != 0 and no tests ran, code didn't compile
        code_compiles = (
            result.exit_code == 0  # Clean execution
            or tests_passed > 0  # Some tests passed (code must have compiled)
            or tests_failed > 0  # Some tests failed (code compiled but tests failed)
        )

        # If no tests detected and non-zero exit, check for compilation errors
        if not code_compiles and tests_passed == 0 and tests_failed == 0:
            # Check stderr for compilation errors
            stderr_lower = result.stderr.lower()
            if any(
                err in stderr_lower
                for err in ["syntaxerror", "indentationerror", "nameerror", "importerror"]
            ):
                code_compiles = False
            else:
                # If no clear compilation error, assume it compiled
                code_compiles = True

        # Calculate reward based on compilation and test results
        reward = self._calculate_reward(code_compiles, tests_passed, tests_failed)

        # Update state
        self._state.step_count += 1
        self._state.last_exit_code = result.exit_code
        self._state.last_code_compiles = code_compiles
        self._state.total_tests_passed = tests_passed
        self._state.total_tests_failed = tests_failed

        # Create observation from execution result
        observation = CodeObservation(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            reward=reward,
            metadata={
                "code": action.code,
                "test_code": action.test_code or "",
            },
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            code_compiles=code_compiles,
        )

        return self._apply_transform(observation)

    def _parse_test_results(self, stdout: str, stderr: str) -> tuple[int, int]:
        """
        Parse Python test output to count passed/failed tests.

        Supports multiple test formats:
        - pytest output: "X passed, Y failed"
        - unittest output: "Ran X tests" with "OK" or "FAILED (failures=Y)"
        - Simple assertions: counts AssertionError occurrences

        Args:
            stdout: Standard output from Python execution
            stderr: Standard error from Python execution

        Returns:
            Tuple of (tests_passed, tests_failed)
        """
        passed = 0
        failed = 0
        output = stdout + "\n" + stderr

        # Method 1: pytest-style output
        # Pattern: "X passed" and "Y failed"
        pytest_passed = re.search(r"(\d+)\s+passed", output)
        pytest_failed = re.search(r"(\d+)\s+failed", output)

        if pytest_passed or pytest_failed:
            if pytest_passed:
                passed = int(pytest_passed.group(1))
            if pytest_failed:
                failed = int(pytest_failed.group(1))
            return passed, failed

        # Method 2: unittest-style output
        # Pattern: "Ran X test(s)" with "OK" or "FAILED (failures=Y, errors=Z)"
        unittest_ran = re.search(r"Ran\s+(\d+)\s+tests?", output)
        if unittest_ran:
            total_tests = int(unittest_ran.group(1))
            # Check for failures
            failures_match = re.search(r"failures=(\d+)", output)
            errors_match = re.search(r"errors=(\d+)", output)

            failures = int(failures_match.group(1)) if failures_match else 0
            errors = int(errors_match.group(1)) if errors_match else 0
            failed = failures + errors

            if "OK" in output or failed == 0:
                passed = total_tests
            else:
                passed = total_tests - failed

            return passed, failed

        # Method 3: Simple assertion counting
        # Count AssertionError occurrences as failures
        assertion_errors = len(re.findall(r"AssertionError", output))
        if assertion_errors > 0:
            failed = assertion_errors
            return passed, failed

        # Method 4: Look for "assert" statements that passed
        # If code ran without errors and contained assertions, count as passed
        if "assert " in output or "assert(" in output:
            # If no errors, assume assertions passed
            if "Error" not in output and "Traceback" not in output:
                passed = 1

        return passed, failed

    def _calculate_reward(
        self, code_compiles: bool, tests_passed: int, tests_failed: int
    ) -> int:
        """
        Optimized integer reward for Python GRPO.
        Strong signal shaping: rewards correctness, penalizes instability,
        and gives higher incentive for near-perfect results.

        Args:
            code_compiles: Whether the code compiled successfully
            tests_passed: Number of tests that passed
            tests_failed: Number of tests that failed

        Returns:
            Integer reward value
        """
        # Code doesn't compile — immediate strong penalty
        if not code_compiles:
            return -3

        reward = 1

        reward += 3 * tests_passed - 1 * tests_failed

        if tests_failed == 0 and tests_passed > 0:
            reward += 2

        return reward

    def _apply_transform(self, observation: CodeObservation) -> CodeObservation:
        """Apply safety and quality transforms to observation."""
        if self.transform:
            observation = self.transform(observation)
        return observation

    @property
    def state(self) -> CodeState:
        """Get current environment state including last exit code."""
        return self._state
