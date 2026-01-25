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
        "python-dotenv>=1.0.1,<2.0.0",
        "pydantic>=2.10.3,<3.0.0",
        "pydantic-settings>=2.7.0,<3.0.0",
        "openai>=1.57.0,<2.0.0",
        "langgraph>=0.2.45,<0.3.0",
        "langchain>=0.3.13,<0.4.0",
        "langchain-core>=0.3.23,<0.4.0",
        "langchain-openai>=0.2.9,<0.3.0",
        "pandas>=2.2.3,<3.0.0",
        "numpy>=2.2.1,<3.0.0",
        "yfinance>=0.2.50,<0.3.0",
        "requests>=2.32.3,<3.0.0",
        "httpx>=0.28.1,<0.29.0",
        "aiohttp>=3.11.11,<4.0.0",
        "python-dateutil>=2.9.0,<3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.4,<9.0.0",
            "pytest-asyncio>=0.24.0,<0.25.0",
            "pytest-cov>=6.0.0,<7.0.0",
            "black>=24.10.0,<25.0.0",
            "flake8>=7.1.1,<8.0.0",
            "mypy>=1.13.0,<2.0.0",
            "types-requests>=2.32.0,<3.0.0",
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
        "Programming Language :: Python :: 3.13",
    ],
    keywords="stock analysis ai agents langgraph openai finance",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stock-analysis-system/issues",
        "Source": "https://github.com/yourusername/stock-analysis-system",
    },
)
