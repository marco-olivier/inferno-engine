from setuptools import setup, find_packages

setup(
    name="inferno-engine",
    version="0.1.0",
    author="Marco Olivier",
    author_email="marco.olivier@inferno-ml.dev",
    description="High-performance AI/ML inference engine",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "onnxruntime>=1.15.0",
        "grpcio>=1.50.0",
        "protobuf>=4.21.0",
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
        "tensorrt": ["tensorrt>=8.6"],
        "dev": ["pytest", "black", "mypy"],
    },
)
