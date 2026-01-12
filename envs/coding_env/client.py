"""
CodingEnv
---------
Client-side wrapper for the Coding environment server.

This client maintains a persistent WebSocket connection to the environment
server, enabling efficient multi-step interactions with lower latency.

- users instantiate CodingEnv with a base_url provided by the higher-level
  vector/orchestration layer.
- Environment authors ship the Docker image that serves the API.

(Seeds, episode IDs, request IDs, capabilities can be added later in the payloads.)
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import CodeAction, CodeObservation, CodeState


class CodingEnv(EnvClient[CodeAction, CodeObservation, CodeState]):
    """
    WebSocket client for the Python Coding Environment.

    This client connects to a PythonCodeActEnv server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = CodingEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.stdout)
        >>>
        >>> # Execute Python code with tests
        >>> action = CodeAction(
        ...     code='''
        ...     def multiply(a, b):
        ...         return a * b
        ...     ''',
        ...     test_code='''
        ...     assert multiply(3, 4) == 12
        ...     assert multiply(0, 5) == 0
        ...     '''
        ... )
        >>> result = client.step(action)
        >>> print(result.observation.tests_passed)  # 2
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CodingEnv.from_docker_image("coding-env:latest")
        >>> result = client.reset()
        >>> result = client.step(CodeAction(code="print(2 + 2)"))
        >>> print(result.observation.stdout)  # "4\\n"
        >>> client.close()
    """

    # --- HTTPEnvClient abstract hooks ---

    def _step_payload(self, action: CodeAction) -> dict:
        """
        Convert CodeAction to JSON payload for step request.

        Args:
            action: CodeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "code": action.code,
            "test_code": action.test_code,
        }

    def _parse_result(self, payload: dict) -> StepResult[CodeObservation]:
        """
        Parse server response into StepResult[CodeObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with CodeObservation
        """
        obs_data = payload.get("observation", {})
        observation = CodeObservation(
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", 0),
            tests_passed=obs_data.get("tests_passed", 0),
            tests_failed=obs_data.get("tests_failed", 0),
            code_compiles=obs_data.get("code_compiles", True),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult[CodeObservation](
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> CodeState:
        """
        Parse server response into CodeState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            CodeState object with episode_id, step_count, and test tracking fields
        """
        return CodeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            last_exit_code=payload.get("last_exit_code", 0),
            last_code_compiles=payload.get("last_code_compiles", True),
            total_tests_passed=payload.get("total_tests_passed", 0),
            total_tests_failed=payload.get("total_tests_failed", 0),
        )
