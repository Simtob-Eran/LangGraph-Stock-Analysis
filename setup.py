"""Setup script for Stock Analysis System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="stock-analysis-system",
    version="1.0.0",
    description="Multi-Agent AI System for Comprehensive Stock Market Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Research Team",
    author_email="research@example.com",
    url="https://github.com/yourusername/stock-analysis-system",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.3",
        "pydantic-settings>=2.1.0",
        "openai>=1.7.2",
        "langgraph>=0.0.25",
        "langchain>=0.1.0",
        "langchain-core>=0.1.10",
        "pandas>=2.1.4",
        "numpy>=1.26.3",
        "yfinance>=0.2.35",
        "requests>=2.31.0",
        "httpx>=0.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "stock-analysis=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="stock analysis ai agents langgraph openai finance",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stock-analysis-system/issues",
        "Source": "https://github.com/yourusername/stock-analysis-system",
    },
)
