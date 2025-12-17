"""
JuliaEnv
--------
Client-side wrapper for the Julia environment server.
Talks HTTP to a single base_url exposing: /reset and /step.

- users instantiate JuliaEnv with a base_url provided by the higher-level
  vector/orchestration layer.
- Environment authors ship the Docker image that serves the HTTP API.

(Seeds, episode IDs, request IDs, capabilities can be added later in the payloads.)
"""

from __future__ import annotations

from openenv.core.client_types import StepResult

from openenv.core.http_env_client import HTTPEnvClient

from julia_env.models import JuliaAction, JuliaObservation, JuliaState


class JuliaEnv(HTTPEnvClient[JuliaAction, JuliaObservation]):
    # --- HTTPEnvClient abstract hooks ---

    def _step_payload(self, action: JuliaAction) -> dict:
        # Shape expected by the server's /step endpoint under "action"
        return {
            "core_code": action.core_code,
            "test_code": action.test_code,
        }

    def _parse_result(self, payload: dict) -> StepResult[JuliaObservation]:
        # Expecting: { "observation": {...}, "reward": <float|null>, "done": <bool>, "info": {...} }
        obs = JuliaObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> JuliaState:
        """
        Parse server response into JuliaState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            JuliaState object with episode_id, step_count, and execution state
        """
        return JuliaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            last_exit_code=payload.get("last_exit_code", 0),
            last_code_compiles=payload.get("last_code_compiles", True),
            total_tests_passed=payload.get("total_tests_passed", 0),
            total_tests_failed=payload.get("total_tests_failed", 0),
        )
