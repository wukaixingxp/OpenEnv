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

from core.env_server import Action, Environment, Observation
from core.tools import PyExecutor

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
        >>> action = CodeAction(code="def add(a, b): return a + b", test_code="assert add(1, 2) == 3")
        >>> obs = env.step(action)
        >>> print(obs.tests_passed)  # 1
        >>> print(obs.reward)  # Positive reward
    """

    def __init__(
        self,
        additional_imports: list[str] | None = None,
    ):
        self.transform = create_safe_coding_transform()
        self._executor = PyExecutor(additional_imports=additional_imports)
        self._state = CodeState()
        self._additional_imports = additional_imports

    def reset(self) -> Observation:
        """
        Reset environment and start fresh execution session.

        Returns:
            Initial observation with empty stdout/stderr and exit_code=0
        """
        # Initialize fresh state
        self._state = CodeState(episode_id=str(uuid.uuid4()), step_count=0)
        self._state.last_exit_code = 0
        self._state.last_code_compiles = True
        self._state.total_tests_passed = 0
        self._state.total_tests_failed = 0

        # Reset executor to clear any previously defined variables/functions
        self._executor = PyExecutor(additional_imports=self._additional_imports)

        # Reset transform to clear any accumulated state
        self.transform = create_safe_coding_transform()

        # Return initial observation
        observation = CodeObservation(
            stdout="",
            stderr="",
            exit_code=0,
            tests_passed=0,
            tests_failed=0,
            code_compiles=True,
            reward=0.0,
        )

        return self._apply_transform(observation)

    def step(self, action: Action, timeout_s: float | None = None) -> Observation:
        """
        Execute code action and return observation.

        Optimized single-pass execution:
        - Runs code + test_code together with proper test scaffolding
        - Parses test results from structured output
        - Calculates reward based on test success

        Args:
            action: CodeAction containing the code to execute
            timeout_s: Maximum execution time in seconds. If None, uses default timeout (60s).

        Returns:
            CodeObservation with execution results (stdout, stderr, exit_code, test results, reward)

        Raises:
            ValueError: If action is not a CodeAction instance
        """
        if not isinstance(action, CodeAction):
            raise ValueError(f"Expected CodeAction, got {type(action)}")

        # Use default timeout if none provided
        if timeout_s is None:
            timeout_s = 60.0

        # Check if test_code is provided - if not, return reward 0
        if not action.test_code or action.test_code.strip() == "":
            # Execute only the code without tests
            result = self._executor.run(action.code, timeout_s=timeout_s)

            # Check if code compiles
            code_compiles = result.exit_code == 0
            if result.exit_code != 0:
                stderr_lower = result.stderr.lower()
                if any(
                    err in stderr_lower
                    for err in ["syntaxerror", "syntax error", "indentationerror", "nameerror"]
                ):
                    code_compiles = False

            # Update state
            self._state.step_count += 1
            self._state.last_exit_code = result.exit_code
            self._state.last_code_compiles = code_compiles
            self._state.total_tests_passed = 0
            self._state.total_tests_failed = 0

            # Return observation with reward 0 (no tests to validate)
            observation = CodeObservation(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                tests_passed=0,
                tests_failed=0,
                code_compiles=code_compiles,
                reward=0.0,  # No tests means no reward
            )

            return self._apply_transform(observation)

        # Build proper test script with individual test case validation
        test_script = self._build_test_script(action.code, action.test_code)

        # Execute the test script
        result = self._executor.run(test_script, timeout_s=timeout_s)

        # Parse test results from structured output
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

        # If no tests detected and non-zero exit, check for syntax errors
        if not code_compiles and tests_passed == 0 and tests_failed == 0:
            stderr_lower = result.stderr.lower()
            if any(
                err in stderr_lower
                for err in ["syntaxerror", "syntax error", "indentationerror", "nameerror"]
            ):
                code_compiles = False
            else:
                # If no clear syntax error, assume it compiled
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
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            code_compiles=code_compiles,
            reward=reward,
        )

        return self._apply_transform(observation)

    def _build_test_script(self, code: str, test_code: str) -> str:
        """
        Build a proper test script with individual test case validation.

        This follows the GroundTruthTestReward pattern to:
        - Add common imports (matching authorized imports from config)
        - Include user's code
        - Wrap each test case in try/except for individual validation
        - Print structured output for parsing

        Args:
            code: User's code to test
            test_code: Test cases (one per line or as assertions)

        Returns:
            Complete test script ready for execution
        """
        # Common imports that match the authorized additional_imports
        # This should be kept in sync with the YAML config additional_imports
        common_imports = """import math
import re
import sys
import os
import random
import itertools
import functools
import collections
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from collections import defaultdict, Counter, deque
import time
import datetime
import statistics
# Additional imports that may be authorized
try:
    import numpy
except ImportError:
    pass
try:
    import pandas
except ImportError:
    pass
"""

        # Parse test_code into individual test cases
        # Split by newlines and filter out empty lines
        test_cases = [
            line.strip()
            for line in test_code.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]

        # Build the test script
        test_script = f"""{common_imports}

{code}

# Ground truth test cases validation
passed = 0
total = {len(test_cases)}
failed_tests = []

"""

        # Add each test case with proper error handling
        for i, test_case in enumerate(test_cases):
            test_num = i + 1
            test_script += f"""try:
    {test_case}
    passed += 1
    print("Test {test_num} PASSED")
except Exception as e:
    failed_tests.append("Test {test_num} FAILED: " + str(e))
    print("Test {test_num} FAILED: " + str(e))

"""

        # Add summary output
        test_script += """success_rate = passed / total if total > 0 else 0.0
print("PASSED:" + str(passed))
print("TOTAL:" + str(total))
print("SUCCESS_RATE:" + str(success_rate))

if failed_tests:
    print("FAILED_TESTS:")
    for failed in failed_tests[:3]:  # Show first 3 failures
        print("  " + failed)
"""

        return test_script

    def _parse_test_results(self, stdout: str, stderr: str) -> tuple[int, int]:
        """
        Parse Python test output to count passed/failed tests.

        First looks for structured output from our test script:
        - "PASSED:X"
        - "TOTAL:Y"

        Falls back to other test framework patterns if structured output not found.

        Args:
            stdout: Standard output from Python execution
            stderr: Standard error from Python execution

        Returns:
            Tuple of (tests_passed, tests_failed)
        """
        passed = 0
        failed = 0
        output = stdout + "\n" + stderr

        # Method 1: Parse our structured output (most reliable)
        passed_match = re.search(r"PASSED:(\d+)", output)
        total_match = re.search(r"TOTAL:(\d+)", output)

        if passed_match and total_match:
            passed = int(passed_match.group(1))
            total = int(total_match.group(1))
            failed = total - passed
            return passed, failed

        # Method 2: Check for explicit test framework output (unittest, pytest)
        # unittest: "Ran N tests in X.XXs\n\nOK" or "FAILED (failures=N)"
        unittest_pattern = r"Ran\s+(\d+)\s+test"
        match = re.search(unittest_pattern, output)
        if match:
            total_tests = int(match.group(1))
            # Check for failures
            fail_pattern = r"FAILED\s*\((?:failures|errors)=(\d+)\)"
            fail_match = re.search(fail_pattern, output)
            if fail_match:
                failed = int(fail_match.group(1))
                passed = total_tests - failed
            elif "OK" in output:
                passed = total_tests
                failed = 0
            return passed, failed

        # Method 3: pytest output
        # "N passed in X.XXs" or "N failed, M passed in X.XXs"
        pytest_pass_pattern = r"(\d+)\s+passed"
        pytest_fail_pattern = r"(\d+)\s+failed"

        pass_match = re.search(pytest_pass_pattern, output)
        fail_match = re.search(pytest_fail_pattern, output)

        if pass_match or fail_match:
            if pass_match:
                passed = int(pass_match.group(1))
            if fail_match:
                failed = int(fail_match.group(1))
            return passed, failed

        # Method 4: Count individual test output from our test script
        # Look for "Test N PASSED" or "Test N FAILED" patterns
        passed_tests = re.findall(r"Test \d+ PASSED", output)
        failed_tests = re.findall(r"Test \d+ FAILED", output)

        if passed_tests or failed_tests:
            passed = len(passed_tests)
            failed = len(failed_tests)
            return passed, failed

        # Method 5: Count individual assertions (fallback)
        # If test_code contains explicit assertions, count them
        # Look for AssertionError in stderr to count failures
        assertion_errors = re.findall(r"AssertionError", stderr)
        failed = len(assertion_errors)

        # If exit code is 0 and there are no assertion errors,
        # assume all assertions passed
        # Count potential assertions in the output by looking for "assert" statements
        # This is a heuristic - we count how many times test validation happened
        if "assert" in output.lower() and failed == 0:
            # Simple heuristic: if code ran successfully and had assertions,
            # assume 1 test passed (we can't know exact count without test framework)
            passed = 1
        elif failed == 0 and "assert" not in output.lower():
            # No assertions found
            passed = 0
            failed = 0

        return passed, failed

    def _calculate_reward(
        self, code_compiles: bool, tests_passed: int, tests_failed: int
    ) -> float:
        """
        Calculate reward based on code compilation and test results.

        Reward structure (matching Julia environment):
        - Code doesn't compile: -3
        - Code compiles: +1
        - Each test passed: +3
        - Each test failed: -1
        - All tests passed (and at least 1 test): +2 bonus

        Returns:
            Integer reward value
        """
        # Code doesn't compile — immediate strong penalty
        if not code_compiles:
            return -3.0

        reward = 1.0

        # Reward for test results
        reward += 3.0 * tests_passed - 1.0 * tests_failed

        # Bonus for perfect test pass
        if tests_failed == 0 and tests_passed > 0:
            reward += 2.0

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
