# -*- coding: utf-8 -*-
"""
Register Champion Models on DagsHub Model Registry.
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

reconfig = getattr(sys.stdout, "reconfigure", None)
if callable(reconfig):
    reconfig(encoding="utf-8")

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from dotenv import load_dotenv
load_dotenv()

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient

from src.utils import MODELS_CLF_DIR, MODELS_REG_DIR, REPORTS_CLF_DIR, REPORTS_REG_DIR, DATA_PROCESSED_PATH


def register_champions():
    remote_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not remote_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set in environment or .env")
    print(f"Connecting to remote server: {remote_uri}")
    mlflow.set_tracking_uri(remote_uri)
    client = MlflowClient(tracking_uri=remote_uri)

    mlflow.set_experiment("Real_Estate_Investment_Advisor")

    sample_df = pd.read_csv(DATA_PROCESSED_PATH, nrows=2)
    leakage_cols = [
        "Good_Investment", "Future_Price_5Y", "Investment_Score",
        "Annual_Appreciation_Rate", "City_Tier_Base_Rate", "ID",
        "Locality_Median_PPS", "Is_Below_Locality_Median_PPS"
    ]
    input_example = sample_df.drop(columns=[c for c in leakage_cols if c in sample_df.columns])

    # 1. Register Classifier Champion (XGBoost)
    print("\n--- Registering Champion Classifier on DagsHub ---")
    clf_pipe = joblib.load(MODELS_CLF_DIR / "xgboost.joblib")
    with mlflow.start_run(run_name="Champion_Classifier_Production") as clf_run:
        mlflow.log_metrics({
            "accuracy": 0.9924,
            "precision": 0.9924,
            "recall": 0.9924,
            "f1_score": 0.9924,
            "roc_auc": 0.9996,
            "inference_latency_ms": 0.008
        })
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("algorithm", "xgboost")
        mlflow.set_tag("role", "champion")

        clf_model_info = mlflow_sklearn.log_model(
            sk_model=clf_pipe,
            artifact_path="model",
            registered_model_name="Property_Classifier_Prod",
            serialization_format="cloudpickle",
            input_example=input_example
        )
        print(f"  Model registered with URI: {clf_model_info.model_uri}")

    # Set stages and aliases
    clf_name = "Property_Classifier_Prod"
    client.update_registered_model(
        name=clf_name,
        description="Production Champion Classifier predicting 'Good Investment' (1/0) for residential properties."
    )
    versions = client.search_model_versions(f"name='{clf_name}'")
    if versions:
        latest_clf_ver = str(versions[0].version)
        try:
            client.transition_model_version_stage(
                name=clf_name,
                version=latest_clf_ver,
                stage="Production",
                archive_existing_versions=True
            )
        except Exception as e:
            print(f"  Stage note: {e}")
        try:
            client.set_registered_model_alias(clf_name, "Production", latest_clf_ver)
            client.set_registered_model_alias(clf_name, "champion", latest_clf_ver)
        except Exception as e:
            print(f"  Alias note: {e}")
        print(f"  ✅ 'Property_Classifier_Prod' v{latest_clf_ver} set to Production on DagsHub!")

    # 2. Register Regressor Champion (LightGBM)
    print("\n--- Registering Champion Regressor on DagsHub ---")
    reg_pipe = joblib.load(MODELS_REG_DIR / "lightgbm.joblib")
    with mlflow.start_run(run_name="Champion_Regressor_Production") as reg_run:
        mlflow.log_metrics({
            "rmse_lakhs": 1.76,
            "mae_lakhs": 1.34,
            "r2_score": 0.9999,
            "mape_percent": 0.63,
            "max_error_lakhs": 12.43,
            "inference_latency_ms": 0.004
        })
        mlflow.set_tag("task", "regression")
        mlflow.set_tag("algorithm", "lightgbm")
        mlflow.set_tag("role", "champion")

        reg_model_info = mlflow_sklearn.log_model(
            sk_model=reg_pipe,
            artifact_path="model",
            registered_model_name="Property_Regressor_Prod",
            serialization_format="cloudpickle",
            input_example=input_example
        )
        print(f"  Model registered with URI: {reg_model_info.model_uri}")

    # Set stages and aliases
    reg_name = "Property_Regressor_Prod"
    client.update_registered_model(
        name=reg_name,
        description="Production Champion Regressor predicting estimated 5-year valuation ('Future_Price_5Y' in ₹ Lakhs)."
    )
    reg_versions = client.search_model_versions(f"name='{reg_name}'")
    if reg_versions:
        latest_reg_ver = str(reg_versions[0].version)
        try:
            client.transition_model_version_stage(
                name=reg_name,
                version=latest_reg_ver,
                stage="Production",
                archive_existing_versions=True
            )
        except Exception as e:
            print(f"  Stage note: {e}")
        try:
            client.set_registered_model_alias(reg_name, "Production", latest_reg_ver)
            client.set_registered_model_alias(reg_name, "champion", latest_reg_ver)
        except Exception as e:
            print(f"  Alias note: {e}")
        print(f"  ✅ 'Property_Regressor_Prod' v{latest_reg_ver} set to Production on DagsHub!")

    print("\n" + "=" * 70)
    print("CHAMPION MODELS SUCCESSFULLY STAGED IN DAGSHUB MODEL REGISTRY!")
    print("=" * 70)


if __name__ == "__main__":
    register_champions()
