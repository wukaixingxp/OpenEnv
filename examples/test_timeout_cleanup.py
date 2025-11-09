#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script to verify timeout cleanup behavior.

This script demonstrates that when a container times out during startup,
it is automatically cleaned up (stopped and removed).
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs import AutoEnv


def count_running_containers(image_prefix="coding-env"):
    """Count how many containers with the given prefix are running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={image_prefix}", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        containers = [line for line in result.stdout.strip().split("\n") if line]
        return len(containers), containers
    except Exception:
        return -1, []


def main():
    print("=" * 70)
    print("Testing Timeout Cleanup Behavior")
    print("=" * 70)
    print()

    # Check initial container count
    initial_count, initial_containers = count_running_containers()
    print(f"Initial running containers: {initial_count}")
    if initial_containers:
        print(f"  Container IDs: {', '.join(initial_containers)}")
    print()

    # Try to create environment with very short timeout (should fail)
    print("Attempting to create environment with 1-second timeout...")
    print("(This should timeout and trigger cleanup)")
    print()

    try:
        env = AutoEnv.from_docker_image("coding-env:latest", wait_timeout=1.0)
        print("❌ Unexpected: Environment created successfully!")
        env.close()
    except TimeoutError as e:
        print("✓ Got expected TimeoutError:")
        print(f"  {str(e)[:200]}...")
        print()

    # Check container count after timeout
    print("Checking containers after timeout...")
    import time

    time.sleep(2)  # Give Docker time to cleanup

    final_count, final_containers = count_running_containers()
    print(f"Final running containers: {final_count}")
    if final_containers:
        print(f"  Container IDs: {', '.join(final_containers)}")
    print()

    # Verify cleanup
    if final_count == initial_count:
        print("✅ SUCCESS: Container was cleaned up automatically!")
        print("   No orphaned containers left behind.")
    else:
        print("⚠️  WARNING: Container count changed unexpectedly")
        print(f"   Initial: {initial_count}, Final: {final_count}")
        if final_count > initial_count:
            new_containers = set(final_containers) - set(initial_containers)
            print(f"   New containers: {', '.join(new_containers)}")
            print()
            print("   Cleaning up manually...")
            for container_id in new_containers:
                try:
                    subprocess.run(["docker", "stop", container_id], timeout=10)
                    subprocess.run(["docker", "rm", container_id], timeout=10)
                    print(f"   ✓ Cleaned up {container_id}")
                except Exception as e:
                    print(f"   ✗ Failed to cleanup {container_id}: {e}")

    print()
    print("=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
