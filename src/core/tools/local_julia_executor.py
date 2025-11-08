# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local Julia Executor.

This module provides functionality for executing Julia code locally using
subprocess, similar to PyExecutor.

Features:
- Proper process cleanup on timeout (no zombie processes)
- Robust error handling and logging
- Process group management for complete cleanup
- Automatic retry on transient failures
"""

import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from core.env_server.types import CodeExecResult

# Setup logging
logger = logging.getLogger(__name__)


class JuliaExecutor:
    """
    Executor for running Julia code in a subprocess with robust process management.

    This class provides a safe interface to execute Julia code in isolation
    and capture the results including stdout, stderr, and exit code.

    Features:
    - Proper timeout handling without zombie processes
    - Process group cleanup for nested processes
    - Automatic retry on transient failures
    - Comprehensive logging for debugging

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

    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 1,
        use_optimization_flags: bool = True,
    ):
        """
        Initialize the JuliaExecutor.

        Args:
            timeout: Maximum execution time in seconds (default: 60)
            max_retries: Number of retry attempts on transient failures (default: 1)
            use_optimization_flags: Enable Julia performance flags (default: True)

        Raises:
            RuntimeError: If Julia executable is not found in PATH
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_optimization_flags = use_optimization_flags

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

        # Build optimized Julia command with performance flags
        self.base_cmd = [self.julia_path]

        if self.use_optimization_flags:
            # Performance optimization flags:
            # --compile=min: Reduce compilation overhead (faster startup)
            # --optimize=2: Medium optimization level (good balance)
            # --startup-file=no: Don't load ~/.julia/config/startup.jl
            # --history-file=no: Don't save REPL history
            self.base_cmd.extend(
                [
                    "--compile=min",  # Minimize compilation for faster startup
                    "--optimize=2",  # Good optimization level
                    "--startup-file=no",  # Skip startup file
                    "--history-file=no",  # Skip history
                ]
            )

            logger.info("Julia optimization flags enabled for faster execution")

        logger.info(f"JuliaExecutor initialized with Julia at: {self.julia_path}")
        logger.info(f"Command: {' '.join(self.base_cmd)}")
        logger.info(f"Timeout: {self.timeout}s, Max retries: {self.max_retries}")

    def _kill_process_tree(
        self, proc: subprocess.Popen, script_file: Optional[str] = None
    ) -> None:
        """
        Terminate a process and all its children.

        Args:
            proc: The subprocess.Popen instance to terminate
            script_file: Optional script file path to kill if process is stuck
        """
        if proc.poll() is None:  # Process is still running
            try:
                # Try graceful termination first
                logger.warning(f"Terminating process {proc.pid} gracefully...")
                proc.terminate()

                # Wait up to 2 seconds for graceful termination
                try:
                    proc.wait(timeout=2.0)
                    logger.info(f"Process {proc.pid} terminated gracefully")
                    return
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Process {proc.pid} did not terminate, forcing kill..."
                    )

                # Force kill if still running
                proc.kill()
                proc.wait(timeout=2.0)
                logger.info(f"Process {proc.pid} killed forcefully")

            except Exception as e:
                logger.error(f"Error killing process {proc.pid}: {e}")

                # Last resort: try killing via process group
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        logger.info(f"Killed process group for {proc.pid}")
                except Exception as pg_error:
                    logger.error(f"Failed to kill process group: {pg_error}")

    def run(self, code: str) -> CodeExecResult:
        """
        Execute Julia code and return the result with robust error handling.

        This method provides:
        - Automatic retry on transient failures
        - Proper timeout handling without zombie processes
        - Process group cleanup for nested processes
        - Comprehensive error logging

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
        code_file = None

        for attempt in range(self.max_retries + 1):
            proc = None

            try:
                # Create temporary file for Julia code
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".jl", delete=False, encoding="utf-8"
                ) as f:
                    f.write(code)
                    code_file = f.name

                script_name = Path(code_file).name
                logger.debug(
                    f"[Attempt {attempt + 1}/{self.max_retries + 1}] Executing Julia script: {script_name}"
                )

                # Start process with Popen for better control
                # Use process group to ensure we can kill all child processes
                start_time = time.time()

                # On Unix systems, use process groups for better cleanup
                kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                }

                # Create new process group on Unix systems
                if hasattr(os, "setpgrp"):
                    kwargs["preexec_fn"] = os.setpgrp

                proc = subprocess.Popen(self.base_cmd + [code_file], **kwargs)

                logger.debug(
                    f"Started Julia process {proc.pid} for script {script_name}"
                )

                # Wait for process with timeout
                try:
                    stdout, stderr = proc.communicate(timeout=self.timeout)
                    exit_code = proc.returncode
                    elapsed = time.time() - start_time

                    logger.debug(
                        f"Julia execution completed in {elapsed:.2f}s (exit code: {exit_code})"
                    )

                    # Clean up temp file
                    try:
                        Path(code_file).unlink()
                    except Exception as cleanup_error:
                        logger.debug(
                            f"Could not delete temp file {code_file}: {cleanup_error}"
                        )

                    return CodeExecResult(
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=exit_code,
                    )

                except subprocess.TimeoutExpired:
                    logger.error(
                        f"Julia execution timed out after {self.timeout}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )

                    # CRITICAL: Kill the process AND all its children to prevent zombies
                    self._kill_process_tree(proc, code_file)

                    # If this was our last retry, return timeout error
                    if attempt >= self.max_retries:
                        logger.error(
                            f"Julia execution failed permanently after {self.max_retries + 1} timeout attempts"
                        )
                        return CodeExecResult(
                            stdout="",
                            stderr=f"Execution timed out after {self.timeout} seconds (tried {self.max_retries + 1} times)",
                            exit_code=-1,
                        )

                    # Wait before retry
                    logger.info(f"Waiting 1s before retry...")
                    time.sleep(1.0)
                    continue

            except FileNotFoundError:
                logger.error(f"Julia executable not found at {self.julia_path}")
                return CodeExecResult(
                    stdout="",
                    stderr=f"Julia executable not found: {self.julia_path}",
                    exit_code=-1,
                )

            except Exception as e:
                logger.error(
                    f"Error executing Julia code (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )

                # Try to kill process if it exists
                if proc is not None and proc.poll() is None:
                    self._kill_process_tree(proc, code_file)

                # If this was our last retry, return error
                if attempt >= self.max_retries:
                    logger.error(
                        f"Julia execution failed permanently after {self.max_retries + 1} attempts"
                    )
                    return CodeExecResult(
                        stdout="",
                        stderr=f"Error executing Julia code: {str(e)}",
                        exit_code=-1,
                    )

                # Wait before retry
                logger.info(f"Waiting 1s before retry...")
                time.sleep(1.0)
                continue

            finally:
                # Always ensure temp file is cleaned up
                if code_file and Path(code_file).exists():
                    try:
                        Path(code_file).unlink()
                        logger.debug(f"Cleaned up temp file: {code_file}")
                    except Exception as cleanup_error:
                        logger.debug(
                            f"Could not delete temp file {code_file}: {cleanup_error}"
                        )

        # Should never reach here, but just in case
        return CodeExecResult(
            stdout="",
            stderr="Unexpected error: all retries exhausted",
            exit_code=-1,
        )
