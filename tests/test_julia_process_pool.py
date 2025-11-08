#!/usr/bin/env python3
"""
Test script for Julia Process Pool implementation.

This script tests the process pool functionality including:
- Basic execution
- Performance comparison with standard execution
- Error handling
- Concurrent execution
"""

import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.tools.local_julia_executor import JuliaExecutor


def test_basic_execution():
    """Test basic Julia code execution."""
    print("\n=== Test 1: Basic Execution ===")

    executor = JuliaExecutor()

    # Simple print test
    result = executor.run('println("Hello, Julia!")')
    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
    assert (
        "Hello, Julia!" in result.stdout
    ), f"Expected output not found: {result.stdout}"
    print("✓ Basic execution works")


def test_process_pool_execution():
    """Test process pool execution."""
    print("\n=== Test 2: Process Pool Execution ===")

    # Enable process pool
    success = JuliaExecutor.enable_process_pool(size=2)
    if not success:
        print("⚠ Process pool not available, skipping test")
        return

    try:
        executor = JuliaExecutor(use_process_pool=True)

        # Test basic execution
        result = executor.run('println("Hello from pool!")')
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
        assert (
            "Hello from pool!" in result.stdout
        ), f"Expected output not found: {result.stdout}"
        print("✓ Process pool execution works")

        # Test multiple executions
        for i in range(5):
            result = executor.run(f"println({i})")
            assert (
                result.exit_code == 0
            ), f"Expected exit code 0, got {result.exit_code}"
            assert (
                str(i) in result.stdout
            ), f"Expected {i} in output, got: {result.stdout}"

        print("✓ Multiple pool executions work")

    finally:
        JuliaExecutor.shutdown_pool()


def test_error_handling():
    """Test error handling in both modes."""
    print("\n=== Test 3: Error Handling ===")

    executor = JuliaExecutor()

    # Test error in standard mode
    result = executor.run('error("Test error")')
    assert result.exit_code != 0, f"Expected non-zero exit code, got {result.exit_code}"
    assert (
        "Test error" in result.stderr or "Test error" in result.stdout
    ), f"Expected error message not found. stdout: {result.stdout}, stderr: {result.stderr}"
    print("✓ Standard mode error handling works")

    # Test error in pool mode
    success = JuliaExecutor.enable_process_pool(size=2)
    if success:
        try:
            executor = JuliaExecutor(use_process_pool=True)
            result = executor.run('error("Test error in pool")')
            assert (
                result.exit_code != 0
            ), f"Expected non-zero exit code, got {result.exit_code}"
            print("✓ Pool mode error handling works")
        finally:
            JuliaExecutor.shutdown_pool()


def test_performance_comparison():
    """Compare performance between standard and pool execution."""
    print("\n=== Test 4: Performance Comparison ===")

    num_iterations = 10
    code = 'println("test")'

    # Test standard execution
    print(f"Running {num_iterations} iterations in standard mode...")
    executor = JuliaExecutor()
    start_time = time.time()

    for _ in range(num_iterations):
        result = executor.run(code)
        assert result.exit_code == 0

    standard_time = time.time() - start_time
    print(
        f"Standard mode: {standard_time:.2f}s ({standard_time/num_iterations:.3f}s per execution)"
    )

    # Test pool execution
    success = JuliaExecutor.enable_process_pool(size=4)
    if not success:
        print("⚠ Process pool not available, skipping performance test")
        return

    try:
        print(f"Running {num_iterations} iterations in pool mode...")
        executor = JuliaExecutor(use_process_pool=True)
        start_time = time.time()

        for _ in range(num_iterations):
            result = executor.run(code)
            assert result.exit_code == 0

        pool_time = time.time() - start_time
        print(
            f"Pool mode: {pool_time:.2f}s ({pool_time/num_iterations:.3f}s per execution)"
        )

        speedup = standard_time / pool_time if pool_time > 0 else 0
        print(f"\n🚀 Speedup: {speedup:.1f}x faster with process pool!")

        if speedup > 5:
            print("✓ Significant speedup achieved")
        else:
            print("⚠ Speedup is lower than expected (may be due to small test size)")

    finally:
        JuliaExecutor.shutdown_pool()


def test_with_test_module():
    """Test execution with Test module (common use case)."""
    print("\n=== Test 5: Test Module Execution ===")

    code = """
    function add(a, b)
        return a + b
    end
    
    using Test
    @test add(2, 3) == 5
    @test add(-1, 1) == 0
    """

    # Test standard mode
    executor = JuliaExecutor()
    result = executor.run(code)
    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"
    print("✓ Standard mode with Test module works")

    # Test pool mode
    success = JuliaExecutor.enable_process_pool(size=2)
    if success:
        try:
            executor = JuliaExecutor(use_process_pool=True)
            result = executor.run(code)
            assert (
                result.exit_code == 0
            ), f"Expected exit code 0, got {result.exit_code}"
            print("✓ Pool mode with Test module works")
        finally:
            JuliaExecutor.shutdown_pool()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Julia Process Pool Test Suite")
    print("=" * 60)

    try:
        test_basic_execution()
        test_process_pool_execution()
        test_error_handling()
        test_with_test_module()
        test_performance_comparison()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
