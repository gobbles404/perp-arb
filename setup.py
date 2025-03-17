# binance_data_pipeline/setup.py
from setuptools import setup, find_packages

setup(
    name="binance_data_pipeline",
    version="0.1.0",
    description="A data pipeline for fetching and processing Binance market data",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "python-binance>=1.0.16",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "binance-fetch=bin.fetch:main",
            "binance-pipeline=bin.pipeline:main",
        ],
    },
    python_requires=">=3.8",
)
