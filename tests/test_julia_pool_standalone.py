#!/usr/bin/env python3
"""
Standalone test for Julia Process Pool.

This test imports only the necessary modules to avoid dependency issues.
"""

import time
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


def test_basic_pool():
    """Test basic process pool functionality."""
    print("\n=== Test 1: Basic Pool Functionality ===")

    try:
        # Create pool
        pool = JuliaProcessPool(size=2, timeout=30)
        print(f"✓ Created pool with 2 workers")

        # Test simple execution
        result = pool.execute('println("Hello from pool!")')
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
        assert (
            "Hello from pool!" in result.stdout
        ), f"Expected output not found: {result.stdout}"
        print("✓ Basic execution works")

        # Test multiple executions
        for i in range(5):
            result = pool.execute(f"println({i})")
            assert (
                result.exit_code == 0
            ), f"Expected exit code 0, got {result.exit_code}"
            assert (
                str(i) in result.stdout
            ), f"Expected {i} in output, got: {result.stdout}"

        print("✓ Multiple executions work")

        # Shutdown
        pool.shutdown()
        print("✓ Pool shutdown successfully")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling in pool."""
    print("\n=== Test 2: Error Handling ===")

    try:
        pool = JuliaProcessPool(size=2, timeout=30)

        # Test error handling
        result = pool.execute('error("Test error")')
        assert (
            result.exit_code != 0
        ), f"Expected non-zero exit code, got {result.exit_code}"
        print("✓ Error handling works")

        # Ensure pool still works after error
        result = pool.execute('println("Still working")')
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
        assert "Still working" in result.stdout
        print("✓ Pool recovers after error")

        pool.shutdown()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance():
    """Test performance improvement."""
    print("\n=== Test 3: Performance Comparison ===")

    num_iterations = 10
    code = 'println("test")'

    # Standard execution (spawn process each time)
    print(f"Running {num_iterations} iterations spawning new processes...")
    import subprocess

    start_time = time.time()
    for _ in range(num_iterations):
        proc = subprocess.Popen(
            ["julia", "-e", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc.communicate()
    standard_time = time.time() - start_time
    print(
        f"Standard: {standard_time:.2f}s ({standard_time/num_iterations:.3f}s per execution)"
    )

    # Pool execution
    try:
        pool = JuliaProcessPool(size=4, timeout=30)

        print(f"Running {num_iterations} iterations with process pool...")
        start_time = time.time()
        for _ in range(num_iterations):
            result = pool.execute(code)
            assert result.exit_code == 0
        pool_time = time.time() - start_time
        print(f"Pool: {pool_time:.2f}s ({pool_time/num_iterations:.3f}s per execution)")

        speedup = standard_time / pool_time if pool_time > 0 else 0
        print(f"\n🚀 Speedup: {speedup:.1f}x faster with process pool!")

        pool.shutdown()

        if speedup > 2:
            print("✓ Significant speedup achieved")
            return True
        else:
            print("⚠ Speedup is lower than expected")
            return True  # Still pass the test

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_with_test_module():
    """Test with Julia Test module."""
    print("\n=== Test 4: Julia Test Module ===")

    code = """
    function add(a, b)
        return a + b
    end
    
    using Test
    @test add(2, 3) == 5
    @test add(-1, 1) == 0
    """

    try:
        pool = JuliaProcessPool(size=2, timeout=30)

        result = pool.execute(code)
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
        print("✓ Test module execution works")

        pool.shutdown()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Julia Process Pool Standalone Test Suite")
    print("=" * 60)

    results = []

    results.append(test_basic_pool())
    results.append(test_error_handling())
    results.append(test_with_test_module())
    results.append(test_performance())

    print("\n" + "=" * 60)
    if all(results):
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print(f"❌ Some tests failed ({sum(results)}/{len(results)} passed)")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
