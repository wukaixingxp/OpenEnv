# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local R Executor.

This module provides functionality for executing R code locally using
subprocess, similar to PyExecutor and JuliaExecutor.
"""

import subprocess
import tempfile
import os
from pathlib import Path

from core.env_server.types import CodeExecResult


class RExecutor:
    """
    Executor for running R code in a subprocess.
    
    This class provides two execution modes:
    1. run() - Basic code execution (compilation/syntax check)
       Executes: Rscript code.R
       
    2. run_with_tests() - Execute code with testthat tests
       Combines core_code + test_code into one file, then executes:
       Rscript -e "testthat::test_file('test.R')"
    
    Example:
        >>> executor = RExecutor()
        >>> 
        >>> # Stage 1: Check if code compiles/runs
        >>> result = executor.run('add <- function(a, b) { a + b }')
        >>> print(result.exit_code)  # 0 means it compiles
        >>> 
        >>> # Stage 2: Run with tests - combines into single file
        >>> core = 'add <- function(a, b) { a + b }'
        >>> tests = '''
        ... library(testthat)
        ... test_that("add works", {
        ...     expect_equal(add(2, 3), 5)
        ... })
        ... '''
        >>> result = executor.run_with_tests(core, tests)
        >>> print(result.exit_code)  # 0
    """
    
    def __init__(self, timeout: int = 60):
        """
        Initialize the RExecutor.
        
        Args:
            timeout: Maximum execution time in seconds (default: 60)
        """
        self.timeout = timeout
    
    def run(self, code: str) -> CodeExecResult:
        """
        Execute R code and return the result (basic execution).
        
        This is used for Stage 1: Compilation/Syntax Check
        Internally runs: Rscript code.R
        
        Args:
            code: R code string to execute
            
        Returns:
            CodeExecResult containing stdout, stderr, and exit_code
            
        Example:
            >>> executor = RExecutor()
            >>> result = executor.run("x <- 5 + 3\\nprint(x)")
            >>> print(result.stdout)  # "[1] 8\n"
            >>> print(result.exit_code)  # 0
            >>>
            >>> # Check if code compiles
            >>> result = executor.run("add <- function(a, b) { a + b }")
            >>> print(result.exit_code)  # 0 means it compiles
        """
        return self._execute_rscript(code)
    
    def run_with_tests(self, core_code: str, test_code: str) -> CodeExecResult:
        """
        Execute R code with testthat tests.
        
        This is used for Stage 2: Test Execution
        Combines core_code and test_code into a single file, then runs:
        Rscript -e "testthat::test_file('test_file.R')"
        
        This triggers testthat's formatted output with the summary box:
        [ FAIL N | WARN W | SKIP S | PASS P ]
        
        Args:
            core_code: Main R code (function definitions, etc.)
            test_code: Test code using testthat
            
        Returns:
            CodeExecResult containing stdout, stderr, and exit_code
            
        Example:
            >>> executor = RExecutor()
            >>> core = '''
            ... add <- function(a, b) {
            ...     return(a + b)
            ... }
            ... '''
            >>> tests = '''
            ... library(testthat)
            ... test_that("add works", {
            ...     expect_equal(add(2, 3), 5)
            ... })
            ... '''
            >>> result = executor.run_with_tests(core, tests)
            >>> print(result.exit_code)  # 0 if tests pass
        """
        try:
            # Combine core code and test code into a single file
            combined_code = core_code + "\n\n" + test_code
            
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.R',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(combined_code)
                test_file = f.name
            
            try:
                test_file_normalized = test_file.replace('\\', '/')
                r_command = f"testthat::test_file('{test_file_normalized}')"
                
                result = subprocess.run(
                    ['Rscript', '-e', r_command],
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
                    Path(test_file).unlink()
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
                stderr=f"Error executing R code with tests: {str(e)}",
                exit_code=-1,
            )
    
    def _execute_rscript(self, code: str) -> CodeExecResult:
        """
        Internal method to execute R code using Rscript.
        
        Args:
            code: R code string to execute
            
        Returns:
            CodeExecResult containing stdout, stderr, and exit_code
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.R',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                code_file = f.name      
            try:
                result = subprocess.run(
                    ['Rscript', code_file],
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
                stderr=f"Error executing R code: {str(e)}",
                exit_code=-1,
            )


