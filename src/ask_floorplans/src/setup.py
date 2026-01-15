"""
Setup configuration for Floor Plan AI System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="floorplan-ai-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered floor plan management system with RAG and copilot capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/floorplan-ai-system",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/floorplan-ai-system/issues",
        "Documentation": "https://github.com/yourusername/floorplan-ai-system/wiki",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Architecture",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
        "production": [
            "pinecone-client>=3.0.0",
            "neo4j>=5.15.0",
            "gunicorn>=21.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "floorplan-cli=floorplan_system:main",
        ],
    },
)
