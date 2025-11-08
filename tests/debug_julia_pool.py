#!/usr/bin/env python3
"""
Debug script for Julia Process Pool.
"""

import sys
import importlib.util
from pathlib import Path
from dataclasses import dataclass


# Define CodeExecResult here to avoid import issues
@dataclass
class CodeExecResult:
    """Result of code execution."""

    stdout: str
    stderr: str
    exit_code: int


# Create a fake types module to satisfy imports
class FakeTypesModule:
    CodeExecResult = CodeExecResult


sys.modules["core.env_server.types"] = FakeTypesModule()

# Now import our modules directly without triggering package __init__
pool_file = (
    Path(__file__).parent.parent / "src" / "core" / "tools" / "julia_process_pool.py"
)

spec = importlib.util.spec_from_file_location("julia_process_pool", pool_file)
julia_process_pool = importlib.util.module_from_spec(spec)
sys.modules["julia_process_pool"] = julia_process_pool
spec.loader.exec_module(julia_process_pool)

JuliaProcessPool = julia_process_pool.JuliaProcessPool

# Test basic execution with detailed output
print("Creating pool...")
pool = JuliaProcessPool(size=1, timeout=30)
print("Pool created successfully")

print("\nExecuting simple println...")
result = pool.execute('println("Hello from pool!")')

print(f"\n=== Result ===")
print(f"Exit code: {result.exit_code}")
print(f"Stdout: {repr(result.stdout)}")
print(f"Stderr: {repr(result.stderr)}")

pool.shutdown()
print("\nPool shutdown")
