# Podman Networking Issue - Solution Summary

## Problem
When using `podman-docker` (podman emulating Docker), the container provider was failing with timeout errors:
```
TimeoutError: Container at http://localhost:63915 did not become ready within 100s
```

### Root Cause Analysis
1. **IPv6 Connection Reset**: When connecting to `localhost`, curl was trying IPv6 (`::1`) and getting "Connection reset by peer"
2. **IPv4 Connection Refused**: When connecting to `127.0.0.1`, curl was getting "Connection refused"
3. **Podman Networking Issue**: Rootless podman has known networking issues with port forwarding using `pasta` or `slirp4netns`

```bash
# IPv6 attempt
$ curl http://localhost:63915/health
* Connected to localhost (::1) port 63915
* Recv failure: Connection reset by peer

# IPv4 attempt
$ curl http://127.0.0.1:63915/health
* Failed to connect to 127.0.0.1 port 63915: Connection refused
```

## Solution: PodmanProvider with Host Networking

Created a dedicated `PodmanProvider` class that uses native podman commands with `--network=host` to bypass port forwarding issues.

### Key Implementation Details

**File**: `/home/kaiwu/work/kaiwu/OpenEnv/src/core/containers/runtime/providers.py`

```python
class PodmanProvider(ContainerProvider):
    """
    Container provider for Podman.

    Uses host networking to avoid rootless podman port forwarding issues.
    Container binds directly to port 8000 on the host.
    """

    def start_container(self, image: str, ...) -> str:
        cmd = [
            "podman", "run",
            "-d",
            "--name", self._container_name,
            "--network", "host",  # Host networking bypasses port forwarding
        ]
        # Container binds directly to host port 8000
        return "http://127.0.0.1:8000"
```

**File**: `/home/kaiwu/work/kaiwu/OpenEnv/src/core/containers/runtime/__init__.py`

```python
from .providers import ContainerProvider, KubernetesProvider, LocalDockerProvider, PodmanProvider

__all__ = [
    "ContainerProvider",
    "LocalDockerProvider",
    "PodmanProvider",  # Now exported
    "KubernetesProvider",
]
```

### Usage

**File**: `/home/kaiwu/work/kaiwu/OpenEnv/test_openenv.py`

```python
from openenv_core.containers.runtime import PodmanProvider

# Use PodmanProvider instead of LocalDockerProvider
provider = PodmanProvider()
base_url = provider.start_container("coding-env:latest")
print(base_url)  # http://127.0.0.1:8000

provider.wait_for_ready(base_url, timeout_s=100)
coding_env = CodingEnv(base_url=base_url, provider=provider)
```

## Test Results

```bash
$ python test_openenv.py
http://127.0.0.1:8000
Reset complete: exit_code=0
Code: print('Hello, World!')
  → stdout: Hello, World!
  → exit_code: 0
Code: x = 5 + 3
print(f'Result: {x}')
  → stdout: Result: 8
  → exit_code: 0
Code: import math
print(math.pi)
  → stdout: 3.141592653589793
  → exit_code: 0
```

✅ **All tests passed successfully!**

## Trade-offs and Limitations

### Host Networking Mode
- **Pro**: Bypasses all rootless podman networking issues
- **Pro**: Direct port access (no port forwarding overhead)
- **Con**: Container always uses port 8000 (no dynamic port allocation)
- **Con**: Can only run one container at a time on the same host

### Alternative Approaches Considered

1. **Use explicit IPv4 binding** (`127.0.0.1:port:8000`) - ❌ Failed with pasta error
2. **Use default port mapping** (`port:8000`) - ❌ Same networking issues
3. **Run with root privileges** - ❌ Security concern
4. **Switch to slirp4netns** - ⚠️ More complex, might still have issues

## Recommendations

1. **For local development**: Use `PodmanProvider` - it's simple and reliable
2. **For CI/CD**: Consider using actual Docker or running podman with root
3. **For production**: Use `KubernetesProvider` or cloud-based container services

## When to Use Which Provider

| Provider | Use Case | Networking Mode |
|----------|----------|----------------|
| `LocalDockerProvider` | Docker installed | Port forwarding |
| `PodmanProvider` | Rootless podman | Host networking |
| `KubernetesProvider` | K8s cluster | Service/Ingress |

## Future Improvements

1. Consider adding a `--rootful` option for `PodmanProvider` to enable port forwarding
2. Add dynamic port support by overriding uvicorn command with custom port
3. Create comprehensive documentation on container runtime selection
