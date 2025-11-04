# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local Ruby Executor.

This module provides functionality for executing Ruby code locally using
subprocess, similar to PyExecutor and JuliaExecutor.
"""

import subprocess
import tempfile
import os
from pathlib import Path

from core.env_server.types import CodeExecResult


class RubyExecutor:
    """
    Executor for running Ruby code in a subprocess.
    
    This class provides a simple interface to execute Ruby code in isolation
    and capture the results including stdout, stderr, and exit code.
    
    Example:
        >>> executor = RubyExecutor()
        >>> result = executor.run('puts "Hello, Ruby!"')
        >>> print(result.stdout)  # "Hello, Ruby!\n"
        >>> print(result.exit_code)  # 0
        >>>
        >>> # With tests
        >>> code = '''
        ... def add(a, b)
        ...     a + b
        ... end
        ... 
        ... require 'minitest/autorun'
        ... class TestAdd < Minitest::Test
        ...   def test_add
        ...     assert_equal 5, add(2, 3)
        ...   end
        ... end
        ... '''
        >>> result = executor.run(code)
        >>> print(result.exit_code)  # 0
    """
    
    def __init__(self, timeout: int = 60):
        """
        Initialize the RubyExecutor.
        
        Args:
            timeout: Maximum execution time in seconds (default: 60)
        """
        self.timeout = timeout
    
    def run(self, code: str) -> CodeExecResult:
        """
        Execute Ruby code and return the result.
        
        Args:
            code: Ruby code string to execute
            
        Returns:
            CodeExecResult containing stdout, stderr, and exit_code
            
        Example:
            >>> executor = RubyExecutor()
            >>> result = executor.run("x = 5 + 3\\nputs x")
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
                mode='w',
                suffix='.rb',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                code_file = f.name      
            try:
                result = subprocess.run(
                    ['ruby', code_file],
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
                stderr=f"Error executing Ruby code: {str(e)}",
                exit_code=-1,
            )

