#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cleanup utility for orphaned OpenEnv containers.

This script helps clean up containers that were left running due to
timeouts or other errors before automatic cleanup was implemented.

Usage:
    python examples/cleanup_orphaned_containers.py
    python examples/cleanup_orphaned_containers.py --force
"""

import argparse
import subprocess
import sys


def get_openenv_containers():
    """Get list of running OpenEnv containers."""
    try:
        # Find all containers with common OpenEnv naming patterns
        patterns = [
            "coding-env",
            "echo-env",
            "git-env",
            "atari-env",
            "browsergym-env",
            "chat-env",
            "connect4-env",
            "dipg-env",
            "finrl-env",
            "openspiel-env",
            "sumo-rl-env",
            "textarena-env",
        ]

        all_containers = []
        for pattern in patterns:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name={pattern}",
                    "--format",
                    "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            container_id, name, status = parts[0], parts[1], parts[2]
                            ports = parts[3] if len(parts) > 3 else ""
                            all_containers.append(
                                {
                                    "id": container_id,
                                    "name": name,
                                    "status": status,
                                    "ports": ports,
                                }
                            )

        return all_containers

    except Exception as e:
        print(f"Error getting containers: {e}")
        return []


def cleanup_container(container_id, container_name):
    """Stop and remove a container."""
    try:
        # Stop container
        print(f"  Stopping {container_name}...")
        result = subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            timeout=15,
        )

        if result.returncode != 0:
            print(f"    Warning: Stop failed, trying to remove anyway...")

        # Remove container
        print(f"  Removing {container_name}...")
        result = subprocess.run(
            ["docker", "rm", container_id],
            capture_output=True,
            timeout=10,
        )

        if result.returncode == 0:
            print(f"  ✓ Cleaned up {container_name} ({container_id[:12]})")
            return True
        else:
            print(f"  ✗ Failed to remove {container_name}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout while cleaning up {container_name}")
        return False
    except Exception as e:
        print(f"  ✗ Error cleaning up {container_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup orphaned OpenEnv Docker containers"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation and clean up all found containers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned up without actually doing it",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("OpenEnv Container Cleanup Utility")
    print("=" * 70)
    print()

    # Get containers
    print("Searching for OpenEnv containers...")
    containers = get_openenv_containers()

    if not containers:
        print("✓ No OpenEnv containers found. Nothing to clean up!")
        print()
        return 0

    print(f"Found {len(containers)} OpenEnv container(s):")
    print()

    # Display containers
    for i, container in enumerate(containers, 1):
        print(f"{i}. {container['name']} ({container['id'][:12]})")
        print(f"   Status: {container['status']}")
        if container["ports"]:
            print(f"   Ports: {container['ports']}")
        print()

    # Confirm cleanup
    if args.dry_run:
        print("--dry-run: Would clean up the above containers (not actually doing it)")
        return 0

    if not args.force:
        print("Do you want to clean up these containers? (yes/no): ", end="")
        response = input().strip().lower()
        print()

        if response not in ["yes", "y"]:
            print("Cleanup cancelled.")
            return 0

    # Cleanup containers
    print("Cleaning up containers...")
    print()

    success_count = 0
    for container in containers:
        if cleanup_container(container["id"], container["name"]):
            success_count += 1

    print()
    print("=" * 70)
    print(f"Cleanup complete: {success_count}/{len(containers)} containers cleaned up")
    print("=" * 70)

    return 0 if success_count == len(containers) else 1


if __name__ == "__main__":
    sys.exit(main())
