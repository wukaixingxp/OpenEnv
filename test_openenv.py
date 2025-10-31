from _thread import exit

from envs.coding_env import CodeAction, CodingEnv
from openenv_core.containers.runtime import LocalDockerProvider

provider = LocalDockerProvider()
base_url = provider.start_container("coding-env:latest")
print(base_url)  # http://127.0.0.1:<port>

# Wait for the server to be ready before creating the client
provider.wait_for_ready(base_url, timeout_s=100)

# Use the environment via base_url
# provider.stop_container()
try:
    # Create environment from Docker image
    # coding_env = CodingEnv.from_docker_image("coding-env:latest")
    coding_env = CodingEnv(base_url=base_url, provider=provider)
    # Reset
    result = coding_env.reset()
    print(f"Reset complete: exit_code={result.observation.exit_code}")

    # Execute Python code
    code_samples = [
        "print('Hello, World!')",
        "x = 5 + 3\nprint(f'Result: {x}')",
        "import math\nprint(math.pi)",
    ]

    for code in code_samples:
        result = coding_env.step(CodeAction(code=code))
        print(f"Code: {code}")
        print(f"  → stdout: {result.observation.stdout.strip()}")
        print(f"  → exit_code: {result.observation.exit_code}")

except Exception as e:
    print(f"Error: {e}")

# Always clean up
coding_env.close()
provider.stop_container()
