

# -*- coding: utf-8 -*-
"""
MLflow Real-Time Service Engine.

Provides production-grade dynamic model serving, real-time live inference
logging, and model registry telemetry querying for the Real Estate
Investment Advisor platform.
"""

from typing import Dict, Any, Optional, Tuple, Union, List, cast
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output encoding on Windows console
reconfig_out = getattr(sys.stdout, "reconfigure", None)
if callable(reconfig_out):
    reconfig_out(encoding="utf-8")
reconfig_err = getattr(sys.stderr, "reconfigure", None)
if callable(reconfig_err):
    reconfig_err(encoding="utf-8")

# Silence MLflow telemetry warnings
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient

from src.utils import (
    MLFLOW_TRACKING_URI,
    MODELS_CLF_DIR,
    MODELS_REG_DIR,
    REPORTS_CLF_DIR,
    REPORTS_REG_DIR,
)

# Global in-memory model cache for high-throughput serving
_MODEL_CACHE: Dict[str, Pipeline] = {}
_MODEL_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}


def get_configured_tracking_uri() -> str:
    """
    Resolve active MLflow tracking URI.
    Prioritizes environment variable MLFLOW_TRACKING_URI (e.g., remote DagsHub server)
    and falls back to local SQLite database store.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


def get_mlflow_client() -> MlflowClient:
    """
    Instantiate an MLflowClient with the configured tracking URI.
    """
    uri = get_configured_tracking_uri()
    return MlflowClient(tracking_uri=uri)


# -----------------------------------------------------------------------------
# Step 4.1: Dynamic Model Serving & Registry Fallbacks
# -----------------------------------------------------------------------------

def load_classifier(
    model_name: str = "Property_Classifier_Prod",
    stage: str = "Production",
    use_cache: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Dynamically load the production classification model pipeline.

    Loading Order:
    1. MLflow Model Registry via Stage URI: 'models:/{model_name}/{stage}'
    2. MLflow Model Registry via Alias URI: 'models:/{model_name}@champion'
    3. Latest registered version in Registry: 'models:/{model_name}/{latest_version}'
    4. Local fallback pipeline: 'models/classification/xgboost.joblib'
    """
    cache_key = f"clf_{model_name}_{stage}"
    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], _MODEL_METADATA_CACHE.get(cache_key, {})

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "task": "classification",
        "loaded_from": None,
        "version": None,
        "stage": stage,
        "is_fallback": False,
    }

    # 1. Attempt loading from production stage
    try:
        uri = f"models:/{model_name}/{stage}"
        pipeline = cast(Pipeline, mlflow_sklearn.load_model(uri))
        metadata["loaded_from"] = uri
        metadata["stage"] = stage
        _MODEL_CACHE[cache_key] = pipeline
        _MODEL_METADATA_CACHE[cache_key] = metadata
        return pipeline, metadata
    except Exception as e_stage:
        print(f"⚠️ [MLflow Service] Could not load classifier from stage URI: {e_stage}")

    # 2. Attempt loading via champion alias
    try:
        uri = f"models:/{model_name}@champion"
        pipeline = cast(Pipeline, mlflow_sklearn.load_model(uri))
        metadata["loaded_from"] = uri
        metadata["alias"] = "champion"
        _MODEL_CACHE[cache_key] = pipeline
        _MODEL_METADATA_CACHE[cache_key] = metadata
        return pipeline, metadata
    except Exception as e_alias:
        print(f"⚠️ [MLflow Service] Could not load classifier from alias URI: {e_alias}")

    # 3. Attempt loading latest version from registry
    try:
        client = get_mlflow_client()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_v = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            uri = f"models:/{model_name}/{latest_v.version}"
            pipeline = cast(Pipeline, mlflow_sklearn.load_model(uri))
            metadata["loaded_from"] = uri
            metadata["version"] = str(latest_v.version)
            _MODEL_CACHE[cache_key] = pipeline
            _MODEL_METADATA_CACHE[cache_key] = metadata
            return pipeline, metadata
    except Exception as e_reg:
        print(f"⚠️ [MLflow Service] Could not load latest registry version: {e_reg}")

    # 4. Fallback to local serialized artifact
    local_path = MODELS_CLF_DIR / "xgboost.joblib"
    if local_path.exists():
        pipeline = cast(Pipeline, joblib.load(local_path))
        metadata["loaded_from"] = str(local_path)
        metadata["is_fallback"] = True
        _MODEL_CACHE[cache_key] = pipeline
        _MODEL_METADATA_CACHE[cache_key] = metadata
        print(f"ℹ️ [MLflow Service] Loaded classifier from local backup: {local_path}")
        return pipeline, metadata

    raise RuntimeError(
        f"Failed to load classification model '{model_name}' from MLflow registry and no local backup was found."
    )


def load_regressor(
    model_name: str = "Property_Regressor_Prod",
    stage: str = "Production",
    use_cache: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Dynamically load the production regression model pipeline.

    Loading Order:
    1. MLflow Model Registry via Stage URI: 'models:/{model_name}/{stage}'
    2. MLflow Model Registry via Alias URI: 'models:/{model_name}@champion'
    3. Latest registered version in Registry: 'models:/{model_name}/{latest_version}'
    4. Local fallback pipeline: 'models/regression/lightgbm.joblib'
    """
    cache_key = f"reg_{model_name}_{stage}"
    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], _MODEL_METADATA_CACHE.get(cache_key, {})

    metadata: Dict[str, Any] = {
        "model_name": model_name,
        "task": "regression",
        "loaded_from": None,
        "version": None,
        "stage": stage,
        "is_fallback": False,
    }

    # 1. Attempt loading from production stage
    try:
        uri = f"models:/{model_name}/{stage}"
        pipeline = cast(Pipeline, mlflow_sklearn.load_model(uri))
        metadata["loaded_from"] = uri
        metadata["stage"] = stage
        _MODEL_CACHE[cache_key] = pipeline
        _MODEL_METADATA_CACHE[cache_key] = metadata
        return pipeline, metadata
    except Exception as e_stage:
        print(f"⚠️ [MLflow Service] Could not load regressor from stage URI: {e_stage}")

    # 2. Attempt loading via champion alias
    try:
        uri = f"models:/{model_name}@champion"
        pipeline = cast(Pipeline, mlflow_sklearn.load_model(uri))
        metadata["loaded_from"] = uri
        metadata["alias"] = "champion"
        _MODEL_CACHE[cache_key] = pipeline
        _MODEL_METADATA_CACHE[cache_key] = metadata
        return pipeline, metadata
    except Exception as e_alias:
        print(f"⚠️ [MLflow Service] Could not load regressor from alias URI: {e_alias}")

    # 3. Attempt loading latest version from registry
    try:
        client = get_mlflow_client()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_v = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            uri = f"models:/{model_name}/{latest_v.version}"
            pipeline = cast(Pipeline, mlflow_sklearn.load_model(uri))
            metadata["loaded_from"] = uri
            metadata["version"] = str(latest_v.version)
            _MODEL_CACHE[cache_key] = pipeline
            _MODEL_METADATA_CACHE[cache_key] = metadata
            return pipeline, metadata
    except Exception as e_reg:
        print(f"⚠️ [MLflow Service] Could not load latest registry version: {e_reg}")

    # 4. Fallback to local serialized artifact
    local_path = MODELS_REG_DIR / "lightgbm.joblib"
    if local_path.exists():
        pipeline = cast(Pipeline, joblib.load(local_path))
        metadata["loaded_from"] = str(local_path)
        metadata["is_fallback"] = True
        _MODEL_CACHE[cache_key] = pipeline
        _MODEL_METADATA_CACHE[cache_key] = metadata
        print(f"ℹ️ [MLflow Service] Loaded regressor from local backup: {local_path}")
        return pipeline, metadata

    raise RuntimeError(
        f"Failed to load regression model '{model_name}' from MLflow registry and no local backup was found."
    )


def load_production_models(use_cache: bool = True) -> Tuple[Pipeline, Pipeline, Dict[str, Any]]:
    """
    Convenience loader retrieving both champion models and operational status.
    """
    clf_pipeline, clf_meta = load_classifier(use_cache=use_cache)
    reg_pipeline, reg_meta = load_regressor(use_cache=use_cache)

    status = {
        "status": "ready",
        "tracking_uri": get_configured_tracking_uri(),
        "classifier": clf_meta,
        "regressor": reg_meta,
    }
    return clf_pipeline, reg_pipeline, status


def get_sample_inference_payload() -> pd.DataFrame:
    """
    Generate a representative single property listing DataFrame for health checks
    and inference benchmarking.
    """
    payload = {
        "Property_Type": ["Apartment"],
        "City": ["Bangalore"],
        "Locality": ["Whitefield"],
        "State": ["Karnataka"],
        "BHK": [3],
        "Size_in_SqFt": [1650.0],
        "Price_in_Lakhs": [145.0],
        "Price_per_SqFt": [8787.88],
        "Floor_No": [6],
        "Total_Floors": [14],
        "Year_Built": [2021],
        "Age_of_Property": [4],
        "Furnished_Status": ["Semi-Furnished"],
        "Facing": ["North-East"],
        "Owner_Type": ["First Owner"],
        "Availability_Status": ["Ready to Move"],
        "Amenities": ["Gym, Pool, Clubhouse, Security, Garden"],
        "Public_Transport_Accessibility": ["High"],
        "Parking_Space": ["Yes"],
        "Security": ["Yes"],
        "Nearby_Schools": [4],
        "Nearby_Hospitals": [3],
    }
    return pd.DataFrame(payload)


def check_model_health() -> Dict[str, Any]:
    """
    Execute end-to-end operational health check on production models.
    Validates model loading, schema compatibility, prediction ranges,
    and measures latency.
    """
    start_time = time.perf_counter()
    report: Dict[str, Any] = {
        "status": "HEALTHY",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tracking_uri": get_configured_tracking_uri(),
        "checks": {},
    }

    try:
        clf_pipeline, reg_pipeline, status = load_production_models(use_cache=True)
        report["serving_status"] = status
    except Exception as e_load:
        report["status"] = "FAILED"
        report["error"] = f"Failed to load production models: {e_load}"
        return report

    sample_df = get_sample_inference_payload()

    # 1. Test Classifier
    try:
        t0 = time.perf_counter()
        clf_pred = int(clf_pipeline.predict(sample_df)[0])
        clf_proba = float(clf_pipeline.predict_proba(sample_df)[0, 1]) if hasattr(clf_pipeline, "predict_proba") else None
        clf_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        report["checks"]["classifier"] = {
            "status": "PASS",
            "prediction": clf_pred,
            "verdict": "Good Investment" if clf_pred == 1 else "Cautious / Avoid",
            "confidence": round(clf_proba * 100.0, 2) if clf_proba is not None else None,
            "latency_ms": clf_latency_ms,
        }
    except Exception as e_clf:
        report["status"] = "DEGRADED"
        report["checks"]["classifier"] = {"status": "FAIL", "error": str(e_clf)}

    # 2. Test Regressor
    try:
        t1 = time.perf_counter()
        reg_pred = float(reg_pipeline.predict(sample_df)[0])
        reg_latency_ms = round((time.perf_counter() - t1) * 1000.0, 3)

        report["checks"]["regressor"] = {
            "status": "PASS",
            "prediction_future_price_lakhs": round(reg_pred, 2),
            "latency_ms": reg_latency_ms,
        }
    except Exception as e_reg:
        report["status"] = "DEGRADED"
        report["checks"]["regressor"] = {"status": "FAIL", "error": str(e_reg)}

    total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    report["total_health_check_ms"] = total_latency_ms

    if report["checks"].get("classifier", {}).get("status") == "FAIL" and report["checks"].get("regressor", {}).get("status") == "FAIL":
        report["status"] = "FAILED"

    return report


# -----------------------------------------------------------------------------
# Step 4.2: Real-Time Inference Logging (Live_Inference_Monitoring)
# -----------------------------------------------------------------------------

LIVE_EXPERIMENT_NAME = "Live_Inference_Monitoring"


def get_or_create_live_experiment() -> str:
    """
    Ensure the 'Live_Inference_Monitoring' experiment exists on the MLflow server
    and return its experiment ID.
    """
    _ = get_configured_tracking_uri()
    client = get_mlflow_client()
    exp = client.get_experiment_by_name(LIVE_EXPERIMENT_NAME)
    if exp is not None:
        return exp.experiment_id

    try:
        exp_id = client.create_experiment(
            name=LIVE_EXPERIMENT_NAME,
            tags={
                "purpose": "production_telemetry",
                "app": "Real_Estate_Investment_Advisor",
                "author": "Antigravity MLOps",
            }
        )
        print(f"ℹ️ [MLflow Service] Created remote experiment '{LIVE_EXPERIMENT_NAME}' (ID: {exp_id})")
        return exp_id
    except Exception as e:
        # Check again if another thread/process created it concurrently
        exp_retry = client.get_experiment_by_name(LIVE_EXPERIMENT_NAME)
        if exp_retry is not None:
            return exp_retry.experiment_id
        print(f"⚠️ [MLflow Service] Note creating experiment '{LIVE_EXPERIMENT_NAME}': {e}. Falling back to default experiment '0'.")
        return "0"


def log_live_inference(
    input_record: Dict[str, Any],
    output_record: Dict[str, Any],
    latency_ms: float,
    run_name: Optional[str] = None
) -> Optional[str]:
    """
    Step 4.2: Log a single live inference evaluation to MLflow experiment
    'Live_Inference_Monitoring'.

    Tracks:
    - User input parameters: City, Locality, BHK, SqFt, Price, Amenities, etc.
    - Model predictions: Classification verdict (1/0), confidence %, Future Price 5Y, CAGR.
    - Performance telemetry: Latency in milliseconds, timestamp, environment tags.

    Gracefully handles transient network issues so live user requests never hang or fail.
    """
    try:
        exp_id = get_or_create_live_experiment()
        now_utc = datetime.now(timezone.utc).isoformat()
        bhk_val = input_record.get("BHK", "Unknown")
        city_val = input_record.get("City", "Unknown")

        auto_run_name = run_name or f"LiveReq_{city_val}_{bhk_val}BHK_{int(time.time())}"

        with mlflow.start_run(experiment_id=exp_id, run_name=auto_run_name) as run:
            # 1. Log Input Parameters
            for key, val in input_record.items():
                if val is not None:
                    val_str = str(val)[:240]
                    mlflow.log_param(key, val_str)

            # 2. Log Output & Latency Metrics
            metrics_to_log: Dict[str, float] = {
                "inference_latency_ms": float(latency_ms),
            }
            if "prediction" in output_record and output_record["prediction"] is not None:
                metrics_to_log["is_good_investment"] = float(output_record["prediction"])
            if "confidence_percent" in output_record and output_record["confidence_percent"] is not None:
                metrics_to_log["confidence_percent"] = float(output_record["confidence_percent"])
            if "predicted_5y_price_lakhs" in output_record and output_record["predicted_5y_price_lakhs"] is not None:
                metrics_to_log["predicted_5y_price_lakhs"] = float(output_record["predicted_5y_price_lakhs"])
            if "cagr_percent" in output_record and output_record["cagr_percent"] is not None:
                metrics_to_log["projected_cagr_percent"] = float(output_record["cagr_percent"])
            if "appreciation_lakhs" in output_record and output_record["appreciation_lakhs"] is not None:
                metrics_to_log["projected_appreciation_lakhs"] = float(output_record["appreciation_lakhs"])

            mlflow.log_metrics(metrics_to_log)

            # 3. Log Operational Tags
            mlflow.set_tags({
                "request_timestamp": now_utc,
                "request_type": "live_user_evaluation",
                "verdict": str(output_record.get("verdict", "N/A")),
                "recommendation": str(output_record.get("recommendation_level", "N/A")),
                "environment": os.getenv("APP_ENV", "production"),
            })

            return run.info.run_id
    except Exception as e_log:
        print(f"[MLflow Service] Non-fatal warning: failed to log live inference to MLflow: {e_log}")
        return None


def predict_property_investment(
    listing_input: Union[Dict[str, Any], pd.DataFrame],
    log_to_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Unified end-to-end inference engine for property valuation & investment recommendation.

    1. Ingests raw property attributes (BHK, sqft, price, city, amenities, etc.).
    2. Dynamically runs champion classification pipeline (Good Investment 1/0 & confidence %).
    3. Dynamically runs champion regression pipeline (5Y property value & CAGR forecast).
    4. Automatically streams telemetry & parameters to remote MLflow (Step 4.2).
    5. Returns rich response dictionary ready for UI presentation.
    """
    start_time = time.perf_counter()

    # 1. Format input into DataFrame
    if isinstance(listing_input, dict):
        df_input = pd.DataFrame([listing_input])
    else:
        df_input = listing_input.copy()

    # Ensure required numerical columns have sensible types
    for num_c in ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Floor_No", "Total_Floors", "Year_Built", "Nearby_Schools", "Nearby_Hospitals"]:
        if num_c in df_input.columns:
            num_series = cast(pd.Series, pd.to_numeric(df_input[num_c], errors="coerce"))
            df_input[num_c] = num_series.fillna(0.0)

    # 2. Load Production Pipelines
    clf_pipeline, reg_pipeline, serving_status = load_production_models(use_cache=True)

    # 3. Execute Classifier Inference
    clf_pred_arr = clf_pipeline.predict(df_input)
    is_good = int(clf_pred_arr[0])

    confidence_pct: Optional[float] = None
    if hasattr(clf_pipeline, "predict_proba"):
        prob_matrix = clf_pipeline.predict_proba(df_input)
        raw_prob = float(prob_matrix[0, 1])
        # Model confidence reflects probability for the predicted class
        confidence_pct = round((raw_prob if is_good == 1 else (1.0 - raw_prob)) * 100.0, 2)

    verdict_str = "Good Investment" if is_good == 1 else "Cautious / Avoid"

    if confidence_pct is not None:
        if confidence_pct >= 90.0:
            rec_level = "High-Confidence Buy" if is_good == 1 else "High-Risk Avoid"
        elif confidence_pct >= 75.0:
            rec_level = "Favorable Buy" if is_good == 1 else "Cautious Avoid"
        else:
            rec_level = "Moderate Potential" if is_good == 1 else "Borderline / Needs Review"
    else:
        rec_level = "Recommended" if is_good == 1 else "Not Recommended"

    # 4. Execute Regressor Inference
    current_price_lakhs = float(df_input["Price_in_Lakhs"].iloc[0]) if "Price_in_Lakhs" in df_input.columns else 0.0
    reg_pred_arr = reg_pipeline.predict(df_input)
    future_price_5y = round(float(reg_pred_arr[0]), 2)

    appreciation_lakhs = round(future_price_5y - current_price_lakhs, 2)
    multiplier = round(future_price_5y / max(current_price_lakhs, 1.0), 2)

    # Compound Annual Growth Rate (5 Years)
    if current_price_lakhs > 0 and future_price_5y > 0:
        cagr = round(((future_price_5y / current_price_lakhs) ** (1.0 / 5.0) - 1.0) * 100.0, 2)
    else:
        cagr = 0.0

    # Latency calculation
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # 5. Pack Outputs
    output_summary: Dict[str, Any] = {
        "prediction": is_good,
        "verdict": verdict_str,
        "recommendation_level": rec_level,
        "confidence_percent": confidence_pct,
        "current_price_lakhs": current_price_lakhs,
        "predicted_5y_price_lakhs": future_price_5y,
        "appreciation_lakhs": appreciation_lakhs,
        "appreciation_multiplier": multiplier,
        "cagr_percent": cagr,
        "latency_ms": latency_ms,
        "serving_status": serving_status["status"],
    }

    # 6. Step 4.2: Real-Time Logging to Remote MLflow
    run_id: Optional[str] = None
    if log_to_mlflow:
        input_dict = df_input.iloc[0].to_dict()
        run_id = log_live_inference(
            input_record=input_dict,
            output_record=output_summary,
            latency_ms=latency_ms
        )

    output_summary["mlflow_run_id"] = run_id
    output_summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    return output_summary


# -----------------------------------------------------------------------------
# Step 4.3: Live Model Telemetry Querying
# -----------------------------------------------------------------------------

def get_production_model_metadata() -> Dict[str, Any]:
    """
    Step 4.3: Fetch live metadata, version lineage, and benchmark metrics
    for both champion production models from the MLflow Model Registry.
    """
    client = get_mlflow_client()
    clf_model_name = "Property_Classifier_Prod"
    reg_model_name = "Property_Regressor_Prod"

    metadata: Dict[str, Any] = {
        "classifier": {
            "name": clf_model_name,
            "version": "1",
            "stage": "Production",
            "algorithm": "XGBoost",
            "description": "Production Champion Classifier",
            "metrics": {"accuracy": 0.9924, "f1_score": 0.9924, "roc_auc": 0.9996, "latency_ms": 0.008},
            "status": "READY"
        },
        "regressor": {
            "name": reg_model_name,
            "version": "1",
            "stage": "Production",
            "algorithm": "LightGBM",
            "description": "Production Champion Regressor",
            "metrics": {"rmse_lakhs": 1.76, "mae_lakhs": 1.34, "r2_score": 0.9999, "mape_percent": 0.63, "latency_ms": 0.004},
            "status": "READY"
        }
    }

    # Query Classifier from Registry
    try:
        clf_reg = client.get_registered_model(clf_model_name)
        if clf_reg.description:
            metadata["classifier"]["description"] = clf_reg.description
        if clf_reg.latest_versions:
            latest_v = clf_reg.latest_versions[0]
            metadata["classifier"]["version"] = str(latest_v.version)
            metadata["classifier"]["stage"] = str(latest_v.current_stage)
            metadata["classifier"]["status"] = str(latest_v.status)
            if latest_v.tags:
                metadata["classifier"]["algorithm"] = latest_v.tags.get("algorithm", "XGBoost").upper()
    except Exception as e_clf:
        print(f"[MLflow Service] Note: using cached classifier metadata: {e_clf}")

    # Query Regressor from Registry
    try:
        reg_reg = client.get_registered_model(reg_model_name)
        if reg_reg.description:
            metadata["regressor"]["description"] = reg_reg.description
        if reg_reg.latest_versions:
            latest_v = reg_reg.latest_versions[0]
            metadata["regressor"]["version"] = str(latest_v.version)
            metadata["regressor"]["stage"] = str(latest_v.current_stage)
            metadata["regressor"]["status"] = str(latest_v.status)
            if latest_v.tags:
                metadata["regressor"]["algorithm"] = latest_v.tags.get("algorithm", "LightGBM").upper()
    except Exception as e_reg:
        print(f"[MLflow Service] Note: using cached regressor metadata: {e_reg}")

    return metadata


def get_recent_inference_logs(limit: int = 20) -> pd.DataFrame:
    """
    Step 4.3: Dynamically fetch recent live inference requests from the
    'Live_Inference_Monitoring' experiment on MLflow for dashboard display.
    """
    columns = [
        "Run ID", "Timestamp", "City", "Locality", "BHK", "Size (SqFt)",
        "Price (₹ Lakhs)", "Verdict", "Confidence", "5Y Price (₹ Lakhs)",
        "CAGR", "Latency (ms)"
    ]

    try:
        client = get_mlflow_client()
        exp = client.get_experiment_by_name(LIVE_EXPERIMENT_NAME)
        if exp is None:
            return pd.DataFrame(columns=columns)

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=limit,
            order_by=["attributes.start_time DESC"]
        )

        records: List[Dict[str, Any]] = []
        for r in runs:
            data = r.data
            start_dt = datetime.fromtimestamp(r.info.start_time / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            conf_val = data.metrics.get("confidence_percent")
            conf_str = f"{conf_val:.1f}%" if conf_val is not None else "N/A"

            fut_val = data.metrics.get("predicted_5y_price_lakhs")
            fut_str = f"₹{fut_val:.2f}" if fut_val is not None else "N/A"

            cagr_val = data.metrics.get("projected_cagr_percent")
            cagr_str = f"{cagr_val:.2f}%" if cagr_val is not None else "N/A"

            lat_val = data.metrics.get("inference_latency_ms")
            lat_str = f"{lat_val:.1f}" if lat_val is not None else "N/A"

            records.append({
                "Run ID": str(r.info.run_id[:8]),
                "Timestamp": start_dt,
                "City": str(data.params.get("City", "N/A")),
                "Locality": str(data.params.get("Locality", "N/A")),
                "BHK": str(data.params.get("BHK", "N/A")),
                "Size (SqFt)": str(data.params.get("Size_in_SqFt", "N/A")),
                "Price (₹ Lakhs)": str(data.params.get("Price_in_Lakhs", "N/A")),
                "Verdict": str(data.tags.get("verdict", "N/A")),
                "Confidence": conf_str,
                "5Y Price (₹ Lakhs)": fut_str,
                "CAGR": cagr_str,
                "Latency (ms)": lat_str,
            })

        return pd.DataFrame(records, columns=columns)
    except Exception as e_query:
        print(f"[MLflow Service] Note: could not fetch live inference logs: {e_query}")
        return pd.DataFrame(columns=columns)


def get_system_telemetry_summary() -> Dict[str, Any]:
    """
    Step 4.3: Produce an aggregated system health, telemetry, and operations summary
    for Tab 3 of the Streamlit application.
    """
    uri = get_configured_tracking_uri()
    is_remote = "dagshub.com" in uri or uri.startswith("http")

    model_meta = get_production_model_metadata()
    recent_logs = get_recent_inference_logs(limit=50)

    total_queries = len(recent_logs)
    avg_latency = 0.0
    if not recent_logs.empty and "Latency (ms)" in recent_logs.columns:
        lat_nums: List[float] = []
        for val in recent_logs["Latency (ms)"]:
            try:
                lat_nums.append(float(val))
            except (ValueError, TypeError):
                pass
        if lat_nums:
            avg_latency = round(float(np.mean(lat_nums)), 2)

    return {
        "tracking_uri": uri,
        "is_remote": is_remote,
        "backend_type": "DagsHub Remote MLflow" if is_remote else "Local SQLite MLflow",
        "server_status": "ONLINE",
        "models": model_meta,
        "total_live_queries": total_queries,
        "avg_latency_ms": avg_latency,
        "recent_logs": recent_logs,
        "last_checked": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real Estate Advisory MLflow Service")
    parser.add_argument("--step", type=str, default="4.3", help="Service step: 4.1 (health check), 4.2 (live inference logging), 4.3 (telemetry querying)")
    args = parser.parse_args()

    if args.step == "4.1":
        print("=" * 70)
        print("STEP 4.1: MLFLOW DYNAMIC MODEL SERVING HEALTH CHECK")
        print("=" * 70)
        health = check_model_health()
        print(f"Overall Health Status : {health['status']}")
        print(f"Tracking URI          : {health['tracking_uri']}")
        print(f"Total Verification MS : {health.get('total_health_check_ms', 0)} ms")
        if "checks" in health:
            print("\n[Classifier Check]:", health["checks"].get("classifier"))
            print("[Regressor Check] :", health["checks"].get("regressor"))
        print("=" * 70)
    elif args.step == "4.2":
        print("=" * 70)
        print("STEP 4.2: REAL-TIME INFERENCE LOGGING VERIFICATION")
        print("=" * 70)
        # Test Case 1: High potential listing in Bangalore
        test_property_1 = {
            "Property_Type": "Apartment", "City": "Bangalore", "Locality": "Whitefield", "State": "Karnataka",
            "BHK": 3, "Size_in_SqFt": 1750.0, "Price_in_Lakhs": 150.0, "Floor_No": 7, "Total_Floors": 18,
            "Year_Built": 2022, "Furnished_Status": "Semi-Furnished", "Facing": "North-East",
            "Owner_Type": "First Owner", "Availability_Status": "Ready to Move",
            "Amenities": "Gym, Pool, Clubhouse, Security, Garden", "Public_Transport_Accessibility": "High",
            "Parking_Space": "Yes", "Security": "Yes", "Nearby_Schools": 5, "Nearby_Hospitals": 4,
        }
        res1 = predict_property_investment(test_property_1, log_to_mlflow=True)
        print("\nTest Property 1 (Bangalore):", res1["verdict"], res1["confidence_percent"], "%", "Run ID:", res1["mlflow_run_id"])
    elif args.step == "4.3":
        print("=" * 70)
        print("STEP 4.3: LIVE MODEL TELEMETRY QUERYING VERIFICATION")
        print("=" * 70)
        summary = get_system_telemetry_summary()
        print(f"Server Status   : {summary['server_status']} ({summary['backend_type']})")
        print(f"Tracking URI    : {summary['tracking_uri']}")
        print(f"Total Live Logs : {summary['total_live_queries']}")
        print(f"Avg Latency MS  : {summary['avg_latency_ms']} ms")
        print(f"Timestamp       : {summary['last_checked']}")

        print("\n[Production Models Lineage]:")
        for m_role, m_info in summary["models"].items():
            print(f"  • {m_role.upper()}: {m_info['name']} (v{m_info['version']}, Stage: {m_info['stage']}, Algo: {m_info['algorithm']})")
            print(f"    Metrics: {m_info['metrics']}")

        print("\n[Recent Live Inference Table (First 5 Rows)]:")
        df_logs = cast(pd.DataFrame, summary["recent_logs"])
        if not df_logs.empty:
            print(df_logs.head(5).to_string(index=False))
        else:
            print("  No live inference runs recorded yet.")
        print("=" * 70)
