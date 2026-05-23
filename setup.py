"""
Setup script for Predictive Maintenance & RUL Forecasting Platform.
"""

from setuptools import setup, find_packages

setup(
    name="predictive-maintenance-system",
    version="1.0.0",
    description="Predictive Maintenance & Intelligent RUL Forecasting Platform",
    author="Your Name",
    author_email="your.email@example.com",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.4.0",
        "tensorflow>=2.15.0",
        "plotly>=5.18.0",
        "streamlit>=1.30.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
