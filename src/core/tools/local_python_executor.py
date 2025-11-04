# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local Python Executor.

This module provides functionality for executing Python code locally by wrapping
the smolagents LocalPythonExecutor with timeout protection.
"""

import multiprocessing
import signal
import threading
from typing import Optional

from smolagents import LocalPythonExecutor

from core.env_server.types import CodeExecResult


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Code execution timed out")


def _run_with_timeout(executor, code: str, timeout_s: float, result_container: list):
    """Helper function to run code execution in a separate process with timeout.
    
    Args:
        executor: The LocalPythonExecutor instance
        code: Code to execute
        timeout_s: Timeout in seconds
        result_container: List to store the result (mutated in place)
    """
    try:
        exec_result = executor(code)
        result_container.append({
            'success': True,
            'stdout': exec_result.logs,
            'stderr': '',
            'exit_code': 0
        })
    except Exception as e:
        result_container.append({
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'exit_code': 1
        })


class PyExecutor:
    """
    Wrapper around smolagents LocalPythonExecutor for executing Python code.

    This class provides a simple interface to execute Python code in a subprocess
    and capture the results including stdout, stderr, and exit code.

    Args:
        additional_imports: List of additional module imports to authorize.
                          For example: ["numpy", "pandas", "matplotlib"]
                          These will be added to the base authorized imports.

    Example:
        >>> # Basic usage with default imports
        >>> executor = PyExecutor()
        >>> result = executor.run("print('Hello, World!')")
        >>> print(result.stdout)  # "Hello, World!\n"
        >>> print(result.exit_code)  # 0
        >>>
        >>> # Usage with additional imports
        >>> executor = PyExecutor(additional_imports=["numpy", "pandas"])
        >>> result = executor.run("import numpy as np\\nprint(np.array([1, 2, 3]))")
        >>> print(result.stdout)  # "[1 2 3]\n"
    """

    def __init__(self, additional_imports: list[str] | None = None):
        """
        Initialize the PyExecutor with a LocalPythonExecutor instance.

        Args:
            additional_imports: List of additional module names to authorize for import.
                              Defaults to an empty list if not provided.
        """
        if additional_imports is None:
            additional_imports = []
        self._executor = LocalPythonExecutor(
            additional_authorized_imports=additional_imports
        )
        # Initialize tools to make BASE_PYTHON_TOOLS available (including print)
        self._executor.send_tools({})

    def run(self, code: str, timeout_s: Optional[float] = None) -> CodeExecResult:
        """
        Execute Python code and return the result with optional timeout protection.

        Args:
            code: Python code string to execute
            timeout_s: Maximum execution time in seconds. If None, no timeout is enforced.
                      If the code exceeds this time, it will be terminated with a timeout error.

        Returns:
            CodeExecResult containing stdout, stderr, and exit_code

        Example:
            >>> executor = PyExecutor()
            >>> result = executor.run("x = 5 + 3\\nprint(x)")
            >>> print(result.stdout)  # "8\n"
            >>> print(result.exit_code)  # 0
            >>>
            >>> # Error handling
            >>> result = executor.run("1 / 0")
            >>> print(result.exit_code)  # 1
            >>> print(result.stderr)  # Contains error message
            >>>
            >>> # Timeout protection
            >>> result = executor.run("while True: pass", timeout_s=5.0)
            >>> print(result.exit_code)  # 1
            >>> print("timeout" in result.stderr.lower())  # True
        """
        # Use proper multiprocessing-based timeout for subprocess protection
        if timeout_s is not None and timeout_s > 0:
            return self._run_with_process_timeout(code, timeout_s)
        
        # No timeout - run directly
        try:
            exec_result = self._executor(code)
            return CodeExecResult(
                stdout=exec_result.logs,
                stderr="",
                exit_code=0,
            )
        except Exception as e:
            return CodeExecResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
            )

    def _run_with_process_timeout(self, code: str, timeout_s: float) -> CodeExecResult:
        """Execute code with proper subprocess timeout protection using multiprocessing.
        
        This method uses multiprocessing.Process to isolate code execution and
        ensures the process is properly terminated if it exceeds the timeout.
        
        Args:
            code: Python code to execute
            timeout_s: Timeout in seconds
            
        Returns:
            CodeExecResult with execution results or timeout error
        """
        # Use a Manager to share results between processes
        manager = multiprocessing.Manager()
        result_container = manager.list()
        
        # Create a process to run the code
        process = multiprocessing.Process(
            target=_run_with_timeout,
            args=(self._executor, code, timeout_s, result_container)
        )
        
        try:
            # Start the process
            process.start()
            
            # Wait for completion with timeout
            process.join(timeout=timeout_s)
            
            # Check if process completed
            if process.is_alive():
                # CRITICAL: Process exceeded timeout - KILL IT!
                print(f"WARNING: Code execution timed out after {timeout_s}s, terminating process {process.pid}")
                process.terminate()  # Send SIGTERM
                process.join(timeout=2)  # Wait up to 2s for graceful shutdown
                
                if process.is_alive():
                    # Still alive - force kill
                    print(f"WARNING: Process {process.pid} did not terminate, force killing")
                    process.kill()  # Send SIGKILL
                    process.join(timeout=1)
                
                return CodeExecResult(
                    stdout="",
                    stderr=f"Code execution timed out after {timeout_s} seconds. "
                           f"Process was terminated. Possible infinite loop or extremely long computation.",
                    exit_code=1,
                )
            
            # Process completed - check results
            if result_container:
                result = result_container[0]
                if result['success']:
                    return CodeExecResult(
                        stdout=result['stdout'],
                        stderr=result['stderr'],
                        exit_code=result['exit_code'],
                    )
                else:
                    return CodeExecResult(
                        stdout=result['stdout'],
                        stderr=result['stderr'],
                        exit_code=result['exit_code'],
                    )
            else:
                # Process completed but no result - something went wrong
                return CodeExecResult(
                    stdout="",
                    stderr="Code execution completed but produced no output",
                    exit_code=1,
                )
                
        except Exception as e:
            # Clean up process on exception
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
            
            return CodeExecResult(
                stdout="",
                stderr=f"Error during code execution: {str(e)}",
                exit_code=1,
            )
        finally:
            # Ensure process is cleaned up
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
