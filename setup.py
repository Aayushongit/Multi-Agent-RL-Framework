from setuptools import setup, find_packages

setup(
    name="multi-agent-rl-env",
    version="1.0.0",
    author="Multi-Agent RL Team",
    description="A comprehensive multi-agent reinforcement learning environment framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0", 
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "swig>=4.0.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="reinforcement-learning multi-agent machine-learning ai",
    entry_points={
        'console_scripts': [
            'marl-train=scripts.train:main',
            'marl-benchmark=scripts.benchmark:main',
        ],
    },
)