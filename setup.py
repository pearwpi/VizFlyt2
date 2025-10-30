"""
VizFlyt2 Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="vizflyt2",
    version="0.1.0",
    author="PEAR Lab",
    author_email="",
    description="A flexible perception, dynamics, and planning system for Gaussian Splatting-based simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pearwpi/VizFlyt2",
    packages=find_packages(include=['perception', 'dynamics', 'planning']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "transforms3d>=0.3.1",
        # Note: nerfstudio is optional and needs to be installed separately
        # due to its complex dependencies
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=20.8b1",
            "flake8>=3.8",
        ],
        "viz": [
            "imageio>=2.9.0",
            "imageio[ffmpeg]>=2.9.0",
        ],
        "perception": [
            # Nerfstudio should be installed separately following their guide
            # This is just a placeholder to document the dependency
        ],
    },
    entry_points={
        "console_scripts": [
            # Add any command-line tools here if needed
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.yml", "*.yaml", "*.json"],
    },
)
