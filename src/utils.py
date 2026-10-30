"""
Utility functions for file paths, logging, and common helpers.
"""
from pathlib import Path

# Project root path
ROOT_DIR = Path(__file__).resolve().parent.parent

# Standard data paths
DATA_RAW_PATH = ROOT_DIR / "data" / "raw" / "india_housing_prices.csv"
DATA_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "cleaned_housing_prices.csv"
REPORTS_EDA_DIR = ROOT_DIR / "reports" / "eda"
MLRUNS_DIR = ROOT_DIR / "mlruns"

