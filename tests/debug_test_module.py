#!/usr/bin/env python3
"""
Debug the Test module issue.
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

# Test with Julia Test module
code = """
function add(a, b)
    return a + b
end

using Test
@test add(2, 3) == 5
@test add(-1, 1) == 0
"""

print("Creating pool...")
pool = JuliaProcessPool(size=1, timeout=30)
print("Pool created successfully")

print("\nExecuting Test module code...")
result = pool.execute(code)

print(f"\n=== Result ===")
print(f"Exit code: {result.exit_code}")
print(f"\nStdout:\n{result.stdout}")
print(f"\nStderr:\n{result.stderr}")

pool.shutdown()
print("\nPool shutdown")
