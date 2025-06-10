"""Setup script for PyCronos package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pycronos",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for data analysis and time series processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pycronos",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
        "stats": ["statsmodels>=0.12.0"],
        "plotting": ["plotly>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "pycronos=pycronos.cli:main",
        ],
    },
    keywords="data analysis, time series, statistics, visualization",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pycronos/issues",
        "Source": "https://github.com/yourusername/pycronos",
        "Documentation": "https://pycronos.readthedocs.io/",
    },
)
