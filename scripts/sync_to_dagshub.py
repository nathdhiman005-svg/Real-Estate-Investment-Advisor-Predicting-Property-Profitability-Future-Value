# -*- coding: utf-8 -*-
"""
DagsHub MLflow Sync Script.

Uploads benchmark experiments, leaderboards, artifacts, and champion models
from local training to the remote DagsHub MLflow server.
"""

from typing import Dict, Any, cast
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Reconfigure stdout for unicode safety
reconfig = getattr(sys.stdout, "reconfigure", None)
if callable(reconfig):
    reconfig(encoding="utf-8")

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from dotenv import load_dotenv
load_dotenv()

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient

from src.utils import (
    MODELS_CLF_DIR,
    MODELS_REG_DIR,
    REPORTS_CLF_DIR,
    REPORTS_REG_DIR,
    DATA_PROCESSED_PATH
)


def sync_all_to_dagshub() -> None:
    remote_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not remote_uri or "dagshub.com" not in remote_uri:
        raise ValueError(f"Invalid remote MLFLOW_TRACKING_URI: {remote_uri}")

    print("=" * 70)
    print("SYNCING LOCAL MLFLOW EXPERIMENTS & REGISTRY TO DAGSHUB")
    print(f"Remote Server: {remote_uri}")
    print("=" * 70)

    mlflow.set_tracking_uri(remote_uri)
    client = MlflowClient(tracking_uri=remote_uri)

    exp_name = "Real_Estate_Investment_Advisor"
    mlflow.set_experiment(exp_name)

    # Load leaderboards as list of dictionaries for clean typing
    clf_records = pd.read_csv(REPORTS_CLF_DIR / "classification_leaderboard.csv").to_dict(orient="records")
    reg_records = pd.read_csv(REPORTS_REG_DIR / "regression_leaderboard.csv").to_dict(orient="records")

    # Sample input example for schema logging
    sample_df = pd.read_csv(DATA_PROCESSED_PATH, nrows=2)
    leakage_cols = [
        "Good_Investment", "Future_Price_5Y", "Investment_Score",
        "Annual_Appreciation_Rate", "City_Tier_Base_Rate", "ID",
        "Locality_Median_PPS", "Is_Below_Locality_Median_PPS"
    ]
    input_example = sample_df.drop(columns=[c for c in leakage_cols if c in sample_df.columns])

    # ---------------------------------------------------------
    # 1. Sync Classification Models
    # ---------------------------------------------------------
    print("\n--- Syncing Classification Models ---")
    clf_runs: Dict[str, str] = {}

    for row in clf_records:
        model_name = str(row["Model"])
        joblib_path = MODELS_CLF_DIR / f"{model_name}.joblib"
        if not joblib_path.exists():
            print(f"  ⚠️ Skipping {model_name}: joblib file not found")
            continue

        print(f"  Logging Classification_{model_name}...")
        pipeline = cast(Pipeline, joblib.load(joblib_path))
        report_dir = REPORTS_CLF_DIR / model_name

        with mlflow.start_run(run_name=f"Classification_{model_name}") as run:
            clf_runs[model_name] = run.info.run_id

            # Metrics
            mlflow.log_metrics({
                "accuracy": float(row["Accuracy"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1_score": float(row["F1_Score"]),
                "roc_auc": float(row["ROC_AUC"]),
                "inference_latency_ms": float(row["Inference_Latency_MS"]),
            })

            # Tags
            mlflow.set_tag("task", "classification")
            mlflow.set_tag("model_type", model_name)
            if model_name == "xgboost":
                mlflow.set_tag("role", "champion")

            # Artifacts (Plots)
            if report_dir.exists():
                mlflow.log_artifacts(str(report_dir), artifact_path="evaluation_reports")

            # Model Pipeline
            mlflow_sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
                input_example=input_example
            )
            print(f"    ✅ Logged run {run.info.run_id}")

    # ---------------------------------------------------------
    # 2. Sync Regression Models
    # ---------------------------------------------------------
    print("\n--- Syncing Regression Models ---")
    reg_runs: Dict[str, str] = {}

    for row in reg_records:
        model_name = str(row["Model"])
        joblib_path = MODELS_REG_DIR / f"{model_name}.joblib"
        if not joblib_path.exists():
            print(f"  ⚠️ Skipping {model_name}: joblib file not found")
            continue

        print(f"  Logging Regression_{model_name}...")
        pipeline = cast(Pipeline, joblib.load(joblib_path))
        report_dir = REPORTS_REG_DIR / model_name

        with mlflow.start_run(run_name=f"Regression_{model_name}") as run:
            reg_runs[model_name] = run.info.run_id

            # Metrics
            mlflow.log_metrics({
                "rmse_lakhs": float(row["RMSE_Lakhs"]),
                "mae_lakhs": float(row["MAE_Lakhs"]),
                "r2_score": float(row["R2_Score"]),
                "mape_percent": float(row["MAPE_Percent"]),
                "max_error_lakhs": float(row["Max_Error_Lakhs"]),
                "inference_latency_ms": float(row["Inference_Latency_MS"]),
            })

            # Tags
            mlflow.set_tag("task", "regression")
            mlflow.set_tag("model_type", model_name)
            if model_name == "lightgbm":
                mlflow.set_tag("role", "champion")

            # Artifacts (Plots)
            if report_dir.exists():
                mlflow.log_artifacts(str(report_dir), artifact_path="evaluation_reports")

            # Model Pipeline
            mlflow_sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
                input_example=input_example
            )
            print(f"    ✅ Logged run {run.info.run_id}")

    # ---------------------------------------------------------
    # 3. Register Champion Models on DagsHub Model Registry
    # ---------------------------------------------------------
    print("\n--- Registering Champion Models in DagsHub Registry ---")

    # Classifier Champion: xgboost
    clf_champ_run_id = clf_runs.get("xgboost")
    if clf_champ_run_id:
        clf_model_name = "Property_Classifier_Prod"
        print(f"  Registering '{clf_model_name}' from run {clf_champ_run_id}...")
        mv_clf = mlflow.register_model(
            model_uri=f"runs:/{clf_champ_run_id}/model",
            name=clf_model_name
        )
        clf_ver = str(mv_clf.version)

        client.update_registered_model(
            name=clf_model_name,
            description="Production Champion Classifier predicting 'Good Investment' (1/0) for residential real estate."
        )
        try:
            client.transition_model_version_stage(
                name=clf_model_name,
                version=clf_ver,
                stage="Production",
                archive_existing_versions=True
            )
        except Exception as e:
            print(f"    Note on stage: {e}")

        try:
            client.set_registered_model_alias(clf_model_name, "Production", clf_ver)
            client.set_registered_model_alias(clf_model_name, "champion", clf_ver)
        except Exception as e:
            print(f"    Note on alias: {e}")

        client.set_model_version_tag(clf_model_name, clf_ver, "task", "classification")
        client.set_model_version_tag(clf_model_name, clf_ver, "algorithm", "xgboost")
        client.set_model_version_tag(clf_model_name, clf_ver, "accuracy", "0.9924")
        client.set_model_version_tag(clf_model_name, clf_ver, "f1_score", "0.9924")
        print(f"  ✅ Registered '{clf_model_name}' v{clf_ver} as Production!")

    # Regressor Champion: lightgbm
    reg_champ_run_id = reg_runs.get("lightgbm")
    if reg_champ_run_id:
        reg_model_name = "Property_Regressor_Prod"
        print(f"  Registering '{reg_model_name}' from run {reg_champ_run_id}...")
        mv_reg = mlflow.register_model(
            model_uri=f"runs:/{reg_champ_run_id}/model",
            name=reg_model_name
        )
        reg_ver = str(mv_reg.version)

        client.update_registered_model(
            name=reg_model_name,
            description="Production Champion Regressor predicting estimated 5-year valuation ('Future_Price_5Y' in ₹ Lakhs)."
        )
        try:
            client.transition_model_version_stage(
                name=reg_model_name,
                version=reg_ver,
                stage="Production",
                archive_existing_versions=True
            )
        except Exception as e:
            print(f"    Note on stage: {e}")

        try:
            client.set_registered_model_alias(reg_model_name, "Production", reg_ver)
            client.set_registered_model_alias(reg_model_name, "champion", reg_ver)
        except Exception as e:
            print(f"    Note on alias: {e}")

        client.set_model_version_tag(reg_model_name, reg_ver, "task", "regression")
        client.set_model_version_tag(reg_model_name, reg_ver, "algorithm", "lightgbm")
        client.set_model_version_tag(reg_model_name, reg_ver, "rmse_lakhs", "1.76")
        client.set_model_version_tag(reg_model_name, reg_ver, "r2_score", "0.9999")
        print(f"  ✅ Registered '{reg_model_name}' v{reg_ver} as Production!")

    print("\n" + "=" * 70)
    print("DAGSHUB SYNC COMPLETE! Check your DagsHub MLflow dashboard:")
    print(f"  -> {remote_uri}")
    print("=" * 70)


if __name__ == "__main__":
    sync_all_to_dagshub()
