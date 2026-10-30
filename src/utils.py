# -*- coding: utf-8 -*-
"""
Utility functions for file paths, logging, and common helpers.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Project root path
ROOT_DIR = Path(__file__).resolve().parent.parent

# Standard data paths
DATA_RAW_PATH = ROOT_DIR / "data" / "raw" / "india_housing_prices.csv"
DATA_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "cleaned_housing_prices.csv"

# Reports and visualization paths
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_EDA_DIR = REPORTS_DIR / "eda"
REPORTS_CLF_DIR = REPORTS_DIR / "classification"
REPORTS_REG_DIR = REPORTS_DIR / "regression"

# Models and MLflow paths
MODELS_DIR = ROOT_DIR / "models"
MODELS_CLF_DIR = MODELS_DIR / "classification"
MODELS_REG_DIR = MODELS_DIR / "regression"
MLRUNS_DIR = ROOT_DIR / "mlruns"
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
LOCAL_MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", LOCAL_MLFLOW_TRACKING_URI)
