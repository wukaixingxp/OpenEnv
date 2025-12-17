from setuptools import setup, find_packages

setup(
    name="openenv-julia_env",
    version="0.1.0",
    description="Julia code execution environment for OpenEnv",
    packages=find_packages(),
    install_requires=[
        "openenv-core",
    ],
    python_requires=">=3.10",
)
