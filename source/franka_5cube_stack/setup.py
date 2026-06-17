"""Package setup for franka_5cube_stack Isaac Lab external task extension."""

from setuptools import find_packages, setup

setup(
    name="franka_5cube_stack",
    version="0.1.0",
    description=(
        "Franka 5-cube stacking external task for Isaac Lab — "
        "camera-once PPO reach skill."
    ),
    author="",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[],  # isaaclab, rsl_rl, gymnasium are provided by the host env
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
