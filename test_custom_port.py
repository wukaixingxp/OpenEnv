from envs.coding_env import CodeAction, CodingEnv
from openenv_core.containers.runtime import PodmanProvider

provider = PodmanProvider()

# Test with custom port 9000
base_url = provider.start_container("coding-env:latest", port=9000)
print(f"Container started at: {base_url}")

# Wait for the server to be ready
provider.wait_for_ready(base_url, timeout_s=30)
print("✅ Container is ready!")

# Create environment
coding_env = CodingEnv(base_url=base_url, provider=provider)

# Test with simple code
result = coding_env.reset()
print(f"Reset: exit_code={result.observation.exit_code}")

result = coding_env.step(CodeAction(code="print('Testing custom port 9000!')"))
print(f"Output: {result.observation.stdout.strip()}")

# Cleanup
coding_env.close()
provider.stop_container()
print("✅ Test complete!")
