# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transforms specific to coding environments."""

import ast
import re

from openenv.core.env_server.base_transforms import CompositeTransform
from openenv.core.env_server.interfaces import Transform
from openenv.core.env_server.types import Observation

from coding_env.models import CodeObservation


class CodeSafetyTransform(Transform):
    """Evaluates code safety and assigns penalties for dangerous patterns."""

    def __init__(self, penalty: float = -3.0):
        self.penalty = penalty
        self.dangerous_patterns = [
            r"import\s+os",
            r"import\s+subprocess",
            r"eval\(",
            r"exec\(",
            r"__import__",
            r"open\(",
        ]

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        # Extract last executed code from metadata (supports both keys)
        code = ""
        if observation.metadata:
            code = observation.metadata.get("code", observation.metadata.get("last_code", ""))

        for pattern in self.dangerous_patterns:
            if re.search(pattern, code):
                observation.reward = (observation.reward or 0.0) + self.penalty
                observation.metadata = observation.metadata or {}
                observation.metadata["safety_violation"] = pattern
                return observation

        # Safe code gets neutral reward
        observation.reward = observation.reward or 0.0
        return observation


class CodeQualityTransform(Transform):
    """Evaluates and rewards code quality metrics."""

    def __init__(
        self,
        concise_bonus: float = 1.0,
        max_length_threshold: int = 120,
        syntax_penalty: float = -0.2,
    ):
        self.concise_bonus = concise_bonus
        self.max_length_threshold = max_length_threshold
        self.syntax_penalty = syntax_penalty

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        quality_score = 0.0

        # Extract code from metadata (supports both keys)
        code = ""
        if observation.metadata:
            code = observation.metadata.get("code", observation.metadata.get("last_code", ""))

        if code:
            # Reward concise code
            if len(code.strip()) <= self.max_length_threshold:
                quality_score += self.concise_bonus
            else:
                quality_score -= 0.1  # slight penalty for verbosity

            # Check syntax (redundant but useful for quality assessment)
            try:
                ast.parse(code)
            except SyntaxError:
                quality_score += self.syntax_penalty

        # Add to existing reward
        if observation.reward is None:
            observation.reward = quality_score
        else:
            observation.reward += quality_score

        return observation


def create_safe_coding_transform() -> CompositeTransform:
    """Create a transform focused on safe coding practices and quality."""
    return CompositeTransform([CodeSafetyTransform(), CodeQualityTransform()])
