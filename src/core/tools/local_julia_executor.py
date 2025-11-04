# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local Julia Executor.

This module provides functionality for executing Julia code locally using
subprocess, similar to PyExecutor.
"""

import subprocess
import tempfile
import os
import shutil
from pathlib import Path

from core.env_server.types import CodeExecResult


class JuliaExecutor:
    """
    Executor for running Julia code in a subprocess.

    This class provides a simple interface to execute Julia code in isolation
    and capture the results including stdout, stderr, and exit code.

    Example:
        >>> executor = JuliaExecutor()
        >>> result = executor.run('println("Hello, Julia!")')
        >>> print(result.stdout)  # "Hello, Julia!\n"
        >>> print(result.exit_code)  # 0
        >>>
        >>> # With tests
        >>> code = '''
        ... function add(a, b)
        ...     return a + b
        ... end
        ...
        ... using Test
        ... @test add(2, 3) == 5
        ... '''
        >>> result = executor.run(code)
        >>> print(result.exit_code)  # 0
    """

    def __init__(self, timeout: int = 60):
        """
        Initialize the JuliaExecutor.

        Args:
            timeout: Maximum execution time in seconds (default: 60)

        Raises:
            RuntimeError: If Julia executable is not found in PATH
        """
        self.timeout = timeout

        # Find Julia executable in PATH
        self.julia_path = shutil.which("julia")

        if not self.julia_path:
            # Try common installation paths
            common_paths = [
                os.path.expanduser("~/.juliaup/bin/julia"),
                os.path.expanduser("~/.julia/bin/julia"),
                "/usr/local/bin/julia",
                "/usr/bin/julia",
            ]

            for path in common_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    self.julia_path = path
                    break

        if not self.julia_path:
            raise RuntimeError(
                "Julia executable not found in PATH or common locations. "
                "Please install Julia: https://julialang.org/downloads/ "
                "or ensure it's in your PATH environment variable."
            )

    def run(self, code: str) -> CodeExecResult:
        """
        Execute Julia code and return the result.

        Args:
            code: Julia code string to execute

        Returns:
            CodeExecResult containing stdout, stderr, and exit_code

        Example:
            >>> executor = JuliaExecutor()
            >>> result = executor.run("x = 5 + 3\\nprintln(x)")
            >>> print(result.stdout)  # "8\n"
            >>> print(result.exit_code)  # 0
            >>>
            >>> # Error handling
            >>> result = executor.run("1 / 0")
            >>> print(result.exit_code)  # 1
            >>> print(result.stderr)  # Contains error message
        """

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jl", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                code_file = f.name
            try:
                result = subprocess.run(
                    [self.julia_path, code_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                return CodeExecResult(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.returncode,
                )

            finally:
                try:
                    Path(code_file).unlink()
                except:
                    pass

        except subprocess.TimeoutExpired:
            return CodeExecResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout} seconds",
                exit_code=-1,
            )

        except Exception as e:
            return CodeExecResult(
                stdout="",
                stderr=f"Error executing Julia code: {str(e)}",
                exit_code=-1,
            )
