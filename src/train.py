# -*- coding: utf-8 -*-
"""
Model Training & Benchmarking Engine.

Provides modular functions for training classification and regression models
using end-to-end scikit-learn pipelines with integrated feature engineering,
local report generation, and integrated MLflow experiment tracking.
"""

from typing import Dict, Any, Optional, Tuple, List, cast
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure headless matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Silence MLflow agent telemetry hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient

from src.preprocessing import get_train_test_split, build_pipeline, get_feature_names
from src.utils import MODELS_CLF_DIR, MODELS_REG_DIR, REPORTS_CLF_DIR, REPORTS_REG_DIR, MLFLOW_TRACKING_URI

# Set UTF-8 encoding for standard output if available
reconfigure_fn = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_fn):
    reconfigure_fn(encoding="utf-8")


# -----------------------------------------------------------------------------
# Classification Model Definitions (Step 3.2)
# -----------------------------------------------------------------------------

def get_classification_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate candidate binary classification estimators with tuned
    hyperparameters for the 'Good_Investment' prediction task.
    """
    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=500,
            C=1.0,
            solver="lbfgs",
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            max_samples=0.5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=150,
            max_depth=8,
            num_leaves=31,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        ),
    }
    return models


def train_classification_models(
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    save_models: bool = True,
    save_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Pipeline], Dict[str, float]]:
    """
    Train all candidate classification pipelines on training data (Step 3.2).
    """
    if X_train is None or y_train is None:
        print("Loading training split for 'Good_Investment' classification...")
        X_train, _, y_train, _ = get_train_test_split(target="Good_Investment", test_size=0.2)

    target_dir = save_dir or MODELS_CLF_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    candidate_models = get_classification_models()
    fitted_pipelines: Dict[str, Pipeline] = {}
    training_times: Dict[str, float] = {}

    print(f"\n{'='*70}")
    print(f"STEP 3.2: TRAINING 4 CLASSIFICATION MODELS ({len(X_train):,} listings)")
    print(f"{'='*70}")

    for model_name, estimator in candidate_models.items():
        print(f"\n[Training] {model_name.upper()}...")
        pipeline = build_pipeline(estimator)

        t_start = time.time()
        pipeline.fit(X_train, y_train)
        fit_duration = round(time.time() - t_start, 2)

        fitted_pipelines[model_name] = pipeline
        training_times[model_name] = fit_duration
        print(f"  -> Completed in {fit_duration} seconds.")

        if save_models:
            model_path = target_dir / f"{model_name}.joblib"
            joblib.dump(pipeline, model_path)
            print(f"  -> Saved fitted pipeline to: {model_path}")

    print(f"\n{'='*70}")
    print("ALL 4 CLASSIFICATION MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*70}")
    for name, dur in training_times.items():
        print(f"  - {name:20s}: {dur:.2f}s")

    return fitted_pipelines, training_times


def load_classification_model(model_name: str, models_dir: Optional[Path] = None) -> Pipeline:
    """
    Load a serialized classification pipeline from disk.
    """
    target_dir = models_dir or MODELS_CLF_DIR
    model_path = target_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    loaded_pipeline = joblib.load(model_path)
    return loaded_pipeline


# -----------------------------------------------------------------------------
# Classification Model Evaluation & Visual Reporting (Step 3.3)
# -----------------------------------------------------------------------------

def evaluate_classification_models(
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    models_dict: Optional[Dict[str, Pipeline]] = None,
    save_reports: bool = True,
    log_mlflow: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all 4 candidate classification pipelines on unseen test listings (Step 3.3).
    Generates local visual reports in reports/classification/<model>/ and logs
    all artifacts, parameters, metrics, and models to MLflow.
    """
    if X_test is None or y_test is None:
        print("Loading test split for classification evaluation (50,000 listings)...")
        _, X_test, _, y_test = get_train_test_split(target="Good_Investment", test_size=0.2)

    # Initialize MLflow tracking environment
    if log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Real_Estate_Investment_Advisor")

    REPORTS_CLF_DIR.mkdir(parents=True, exist_ok=True)
    model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    leaderboard_records: List[Dict[str, Any]] = []

    print(f"\n{'='*70}")
    print(f"STEP 3.3: EVALUATING 4 CLASSIFICATION MODELS ON {len(X_test):,} LISTINGS")
    print(f"{'='*70}")

    for model_name in model_names:
        print(f"\n[Evaluating] {model_name.upper()}...")
        if models_dict and model_name in models_dict:
            pipeline = models_dict[model_name]
        else:
            pipeline = load_classification_model(model_name)

        # 1. Predictions & Timing
        t0 = time.time()
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        eval_duration = time.time() - t0
        latency_ms_per_sample = (eval_duration / len(X_test)) * 1000.0

        # 2. Compute Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        ap = float(average_precision_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        record = {
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
            "Average_Precision": round(ap, 4),
            "Inference_Latency_MS": round(latency_ms_per_sample, 3),
            "Total_Inference_Sec": round(eval_duration, 2),
        }
        leaderboard_records.append(record)

        print(f"  -> Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {auc:.4f}")

        # 3. Create Dedicated Model Report Directory
        model_report_dir = REPORTS_CLF_DIR / model_name
        model_report_dir.mkdir(parents=True, exist_ok=True)

        if save_reports:
            # A. Confusion Matrix Plot
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            cm_counts = [f"{val:,}" for val in cm.flatten()]
            cm_pcts = [f"({val/len(y_test):.2%})" for val in cm.flatten()]
            cm_labels = [f"{c}\n{p}" for c, p in zip(cm_counts, cm_pcts)]
            cm_labels_mat = np.array(cm_labels).reshape(2, 2)

            sns.heatmap(
                cm,
                annot=cm_labels_mat,
                fmt="",
                cmap="Blues",
                cbar=False,
                xticklabels=["Average / Poor (0)", "Good Investment (1)"],
                yticklabels=["Average / Poor (0)", "Good Investment (1)"],
                ax=ax_cm,
            )
            ax_cm.set_title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
            ax_cm.set_xlabel("Predicted Class", fontsize=11)
            ax_cm.set_ylabel("Actual Ground Truth", fontsize=11)
            plt.tight_layout()
            cm_path = model_report_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close(fig_cm)

            # B. ROC & Precision-Recall Curves
            fig_curves, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax_roc.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc:.4f})")
            ax_roc.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance")
            ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
            ax_roc.set_xlabel("False Positive Rate", fontsize=10)
            ax_roc.set_ylabel("True Positive Rate", fontsize=10)
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, alpha=0.3)

            prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
            prevalence = float(np.mean(y_test))
            ax_pr.plot(rec_curve, prec_curve, color="#10b981", lw=2, label=f"PR (AP = {ap:.4f})")
            ax_pr.axhline(prevalence, color="#94a3b8", lw=1.5, linestyle="--", label=f"Prevalence ({prevalence:.1%})")
            ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
            ax_pr.set_xlabel("Recall", fontsize=10)
            ax_pr.set_ylabel("Precision", fontsize=10)
            ax_pr.legend(loc="lower left")
            ax_pr.grid(True, alpha=0.3)

            plt.tight_layout()
            curves_path = model_report_dir / "roc_pr_curve.png"
            plt.savefig(curves_path, dpi=150)
            plt.close(fig_curves)

            # C. Top Feature Importances Plot
            feature_names = get_feature_names(pipeline)
            estimator = pipeline.named_steps["model"]
            if hasattr(estimator, "feature_importances_"):
                raw_importances = np.asarray(estimator.feature_importances_, dtype=float)
            elif hasattr(estimator, "coef_"):
                raw_importances = np.asarray(np.abs(estimator.coef_[0]), dtype=float)
            else:
                raw_importances = np.zeros(len(feature_names))

            fi_df = pd.DataFrame({"feature": feature_names, "importance": raw_importances})
            fi_df = fi_df.sort_values(by="importance", ascending=False).head(15)

            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            sns.barplot(data=fi_df, x="importance", y="feature", hue="feature", palette="viridis", legend=False, ax=ax_fi)
            ax_fi.set_title(f"Top 15 Feature Importances: {model_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
            ax_fi.set_xlabel("Relative Importance / Absolute Coefficient", fontsize=11)
            ax_fi.set_ylabel("Feature", fontsize=11)
            plt.tight_layout()
            fi_path = model_report_dir / "feature_importance.png"
            plt.savefig(fi_path, dpi=150)
            plt.close(fig_fi)

            # D. Metrics JSON & CSV Reports
            metrics_dict = {
                "model_name": model_name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
                "average_precision": ap,
                "confusion_matrix": {
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                },
                "latency_ms_per_sample": latency_ms_per_sample,
                "total_eval_time_sec": eval_duration,
            }
            json_path = model_report_dir / "evaluation_metrics.json"
            with open(json_path, "w", encoding="utf-8") as f_json:
                json.dump(metrics_dict, f_json, indent=2)

            rep_df = pd.DataFrame(cast(Dict[str, Any], report_dict)).T
            csv_path = model_report_dir / "classification_report.csv"
            rep_df.to_csv(csv_path)

            print(f"  -> Visual artifacts and tables saved to: {model_report_dir}")

        # 4. Integrated MLflow Experiment Tracking
        if log_mlflow:
            with mlflow.start_run(run_name=f"Classification_{model_name}"):
                # Log model parameters
                raw_model = pipeline.named_steps["model"]
                if hasattr(raw_model, "get_params"):
                    params = raw_model.get_params()
                    for p_name, p_val in list(params.items())[:25]:
                        try:
                            mlflow.log_param(p_name, str(p_v)[:250] if (p_v := p_val) is not None else "None")
                        except Exception:
                            pass

                # Log evaluation metrics
                mlflow.log_metrics({
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "roc_auc": auc,
                    "average_precision": ap,
                    "inference_latency_ms": latency_ms_per_sample,
                    "eval_duration_sec": eval_duration,
                })

                # Log visual report artifacts
                if save_reports:
                    mlflow.log_artifacts(str(model_report_dir), artifact_path="evaluation_reports")

                # Log full pipeline model artifact using cloudpickle for custom transformer compatibility
                mlflow_sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                    input_example=X_test.iloc[:3]
                )
                print(f"  -> Logged to MLflow Run: 'Classification_{model_name}'")

    # 5. Build and Save Cross-Model Comparison
    leaderboard_df = pd.DataFrame(leaderboard_records).sort_values(by="F1_Score", ascending=False)
    leaderboard_csv = REPORTS_CLF_DIR / "classification_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False)

    if save_reports:
        fig_comp, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(15, 5))
        melted_df = leaderboard_df.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
            var_name="Metric",
            value_name="Score"
        )
        sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="tab10", ax=ax_c1)
        ax_c1.set_title("Cross-Model Classification Performance", fontsize=13, fontweight="bold")
        ax_c1.set_ylim(0.95, 1.0)
        ax_c1.set_xlabel("Model", fontsize=11)
        ax_c1.set_ylabel("Score", fontsize=11)
        ax_c1.legend(loc="lower right")

        sns.barplot(data=leaderboard_df, x="Model", y="Inference_Latency_MS", hue="Model", palette="Blues_r", legend=False, ax=ax_c2)
        ax_c2.set_title("Inference Latency (Milliseconds per Sample)", fontsize=13, fontweight="bold")
        ax_c2.set_xlabel("Model", fontsize=11)
        ax_c2.set_ylabel("Latency (ms)", fontsize=11)
        for i, model_name in enumerate(leaderboard_df["Model"]):
            lat_val = float(leaderboard_df["Inference_Latency_MS"].iloc[i])
            ax_c2.text(i, lat_val + 0.0005, f"{lat_val:.3f} ms", ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        comp_img_path = REPORTS_CLF_DIR / "classification_comparison.png"
        plt.savefig(comp_img_path, dpi=150)
        plt.close(fig_comp)
        print(f"\n✅ Cross-model comparison saved to: {comp_img_path}")

    print(f"\n{'='*70}")
    print("STEP 3.3 CLASSIFICATION LEADERBOARD SUMMARY")
    print(f"{'='*70}")
    print(leaderboard_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Inference_Latency_MS"]].to_string(index=False))

    return leaderboard_df


# -----------------------------------------------------------------------------
# Regression Model Definitions & Training (Step 3.4)
# -----------------------------------------------------------------------------

def get_regression_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate candidate regression estimators with tuned hyperparameters
    for the 'Future_Price_5Y' (Lakhs) prediction task.
    """
    models: Dict[str, Any] = {
        "ridge": Ridge(
            alpha=1.0,
            random_state=random_state
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            max_samples=0.5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=150,
            max_depth=8,
            num_leaves=31,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        ),
    }
    return models


def train_regression_models(
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    save_models: bool = True,
    save_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Pipeline], Dict[str, float]]:
    """
    Train all candidate regression pipelines on training data (Step 3.4).
    """
    if X_train is None or y_train is None:
        print("Loading training split for 'Future_Price_5Y' regression...")
        X_train, _, y_train, _ = get_train_test_split(target="Future_Price_5Y", test_size=0.2)

    target_dir = save_dir or MODELS_REG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    candidate_models = get_regression_models()
    fitted_pipelines: Dict[str, Pipeline] = {}
    training_times: Dict[str, float] = {}

    print(f"\n{'='*70}")
    print(f"STEP 3.4: TRAINING 4 REGRESSION MODELS ({len(X_train):,} listings)")
    print(f"{'='*70}")

    for model_name, estimator in candidate_models.items():
        print(f"\n[Training] {model_name.upper()}...")
        pipeline = build_pipeline(estimator)

        t_start = time.time()
        pipeline.fit(X_train, y_train)
        fit_duration = round(time.time() - t_start, 2)

        fitted_pipelines[model_name] = pipeline
        training_times[model_name] = fit_duration
        print(f"  -> Completed in {fit_duration} seconds.")

        if save_models:
            model_path = target_dir / f"{model_name}.joblib"
            joblib.dump(pipeline, model_path)
            print(f"  -> Saved fitted pipeline to: {model_path}")

    print(f"\n{'='*70}")
    print("ALL 4 REGRESSION MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*70}")
    for name, dur in training_times.items():
        print(f"  - {name:20s}: {dur:.2f}s")

    return fitted_pipelines, training_times


def load_regression_model(model_name: str, models_dir: Optional[Path] = None) -> Pipeline:
    """
    Load a serialized regression pipeline from disk.
    """
    target_dir = models_dir or MODELS_REG_DIR
    model_path = target_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    loaded_pipeline = joblib.load(model_path)
    return loaded_pipeline


# -----------------------------------------------------------------------------
# Regression Model Evaluation & Visual Reporting (Step 3.5)
# -----------------------------------------------------------------------------

def evaluate_regression_models(
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    models_dict: Optional[Dict[str, Pipeline]] = None,
    save_reports: bool = True,
    log_mlflow: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all 4 candidate regression pipelines on unseen test listings (Step 3.5).
    Generates local visual reports in reports/regression/<model>/ and logs
    all artifacts, parameters, metrics, and models to MLflow.
    """
    if X_test is None or y_test is None:
        print("Loading test split for regression evaluation (50,000 listings)...")
        _, X_test, _, y_test = get_train_test_split(target="Future_Price_5Y", test_size=0.2)

    # Initialize MLflow tracking environment
    if log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Real_Estate_Investment_Advisor")

    REPORTS_REG_DIR.mkdir(parents=True, exist_ok=True)
    model_names = ["ridge", "random_forest", "xgboost", "lightgbm"]
    leaderboard_records: List[Dict[str, Any]] = []

    print(f"\n{'='*70}")
    print(f"STEP 3.5: EVALUATING 4 REGRESSION MODELS ON {len(X_test):,} LISTINGS")
    print(f"{'='*70}")

    for model_name in model_names:
        print(f"\n[Evaluating] {model_name.upper()}...")
        if models_dict and model_name in models_dict:
            pipeline = models_dict[model_name]
        else:
            pipeline = load_regression_model(model_name)

        # 1. Predictions & Timing
        t0 = time.time()
        y_pred = pipeline.predict(X_test)
        eval_duration = time.time() - t0
        latency_ms_per_sample = (eval_duration / len(X_test)) * 1000.0

        # 2. Compute Metrics
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred)) * 100.0
        max_err = float(np.max(np.abs(y_test - y_pred)))

        record = {
            "Model": model_name,
            "RMSE_Lakhs": round(rmse, 2),
            "MAE_Lakhs": round(mae, 2),
            "R2_Score": round(r2, 4),
            "MAPE_Percent": round(mape, 2),
            "Max_Error_Lakhs": round(max_err, 2),
            "Inference_Latency_MS": round(latency_ms_per_sample, 3),
            "Total_Inference_Sec": round(eval_duration, 2),
        }
        leaderboard_records.append(record)

        print(f"  -> RMSE: ₹{rmse:.2f}L | MAE: ₹{mae:.2f}L | R²: {r2:.4f} | MAPE: {mape:.2f}% | Max Err: ₹{max_err:.2f}L")

        # 3. Create Dedicated Model Report Directory
        model_report_dir = REPORTS_REG_DIR / model_name
        model_report_dir.mkdir(parents=True, exist_ok=True)

        if save_reports:
            # A. Actual vs. Predicted Plot
            fig_avp, ax_avp = plt.subplots(figsize=(8, 6))
            sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(5000, len(y_test)), replace=False)
            actual_sample = np.asarray(y_test)[sample_idx]
            pred_sample = np.asarray(y_pred)[sample_idx]

            ax_avp.scatter(actual_sample, pred_sample, alpha=0.25, color="#2563eb", edgecolors="none", s=20, label="Properties")
            min_val = min(float(np.min(actual_sample)), float(np.min(pred_sample)))
            max_val = max(float(np.max(actual_sample)), float(np.max(pred_sample)))
            ax_avp.plot([min_val, max_val], [min_val, max_val], color="#ef4444", lw=2, linestyle="--", label="Perfect Prediction (y = x)")

            ax_avp.set_title(f"Actual vs. Predicted 5Y Price: {model_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
            ax_avp.set_xlabel("Actual 5-Year Future Price (₹ Lakhs)", fontsize=11)
            ax_avp.set_ylabel("Predicted 5-Year Future Price (₹ Lakhs)", fontsize=11)
            ax_avp.text(
                0.05, 0.85,
                f"R² = {r2:.4f}\nRMSE = ₹{rmse:.2f}L\nMAE = ₹{mae:.2f}L",
                transform=ax_avp.transAxes,
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#94a3b8")
            )
            ax_avp.legend(loc="lower right")
            ax_avp.grid(True, alpha=0.3)
            plt.tight_layout()
            avp_path = model_report_dir / "actual_vs_predicted.png"
            plt.savefig(avp_path, dpi=150)
            plt.close(fig_avp)

            # B. Residuals Distribution Plot
            residuals = y_test - y_pred
            fig_res, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(14, 5))

            sns.histplot(residuals, kde=True, color="#6366f1", bins=50, ax=ax_r1)
            ax_r1.axvline(0, color="#ef4444", linestyle="--", lw=1.5, label="Zero Error")
            ax_r1.set_title(f"Residuals Distribution: {model_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax_r1.set_xlabel("Residual (Actual - Predicted in ₹ Lakhs)", fontsize=10)
            ax_r1.set_ylabel("Listing Count", fontsize=10)
            ax_r1.legend()
            ax_r1.grid(True, alpha=0.3)

            ax_r2.scatter(pred_sample, (actual_sample - pred_sample), alpha=0.25, color="#0ea5e9", edgecolors="none", s=20)
            ax_r2.axhline(0, color="#ef4444", linestyle="--", lw=1.5)
            ax_r2.set_title("Residuals vs. Predicted Price", fontsize=12, fontweight="bold")
            ax_r2.set_xlabel("Predicted 5-Year Price (₹ Lakhs)", fontsize=10)
            ax_r2.set_ylabel("Residual (₹ Lakhs)", fontsize=10)
            ax_r2.grid(True, alpha=0.3)

            plt.tight_layout()
            res_path = model_report_dir / "residuals_distribution.png"
            plt.savefig(res_path, dpi=150)
            plt.close(fig_res)

            # C. Top Feature Importances / Coefficients Plot
            feature_names = get_feature_names(pipeline)
            estimator = pipeline.named_steps["model"]
            if hasattr(estimator, "feature_importances_"):
                raw_importances = np.asarray(estimator.feature_importances_, dtype=float)
            elif hasattr(estimator, "coef_"):
                raw_importances = np.asarray(np.abs(estimator.coef_), dtype=float)
            else:
                raw_importances = np.zeros(len(feature_names))

            fi_df = pd.DataFrame({"feature": feature_names, "importance": raw_importances})
            fi_df = fi_df.sort_values(by="importance", ascending=False).head(15)

            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            sns.barplot(data=fi_df, x="importance", y="feature", hue="feature", palette="magma", legend=False, ax=ax_fi)
            ax_fi.set_title(f"Top 15 Feature Importances: {model_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
            ax_fi.set_xlabel("Relative Importance / Absolute Coefficient", fontsize=11)
            ax_fi.set_ylabel("Feature", fontsize=11)
            plt.tight_layout()
            fi_path = model_report_dir / "feature_importance.png"
            plt.savefig(fi_path, dpi=150)
            plt.close(fig_fi)

            # D. Metrics JSON & CSV Reports
            metrics_dict = {
                "model_name": model_name,
                "rmse_lakhs": rmse,
                "mae_lakhs": mae,
                "r2_score": r2,
                "mape_percent": mape,
                "max_error_lakhs": max_err,
                "latency_ms_per_sample": latency_ms_per_sample,
                "total_eval_time_sec": eval_duration,
            }
            json_path = model_report_dir / "evaluation_metrics.json"
            with open(json_path, "w", encoding="utf-8") as f_json:
                json.dump(metrics_dict, f_json, indent=2)

            rep_df = pd.DataFrame([metrics_dict])
            csv_path = model_report_dir / "regression_summary.csv"
            rep_df.to_csv(csv_path, index=False)

            print(f"  -> Visual artifacts and tables saved to: {model_report_dir}")

        # 4. Integrated MLflow Experiment Tracking
        if log_mlflow:
            with mlflow.start_run(run_name=f"Regression_{model_name}"):
                # Log model parameters
                raw_model = pipeline.named_steps["model"]
                if hasattr(raw_model, "get_params"):
                    params = raw_model.get_params()
                    for p_name, p_val in list(params.items())[:25]:
                        try:
                            mlflow.log_param(p_name, str(p_v)[:250] if (p_v := p_val) is not None else "None")
                        except Exception:
                            pass

                # Log evaluation metrics
                mlflow.log_metrics({
                    "rmse_lakhs": rmse,
                    "mae_lakhs": mae,
                    "r2_score": r2,
                    "mape_percent": mape,
                    "max_error_lakhs": max_err,
                    "inference_latency_ms": latency_ms_per_sample,
                    "eval_duration_sec": eval_duration,
                })

                # Log visual report artifacts
                if save_reports:
                    mlflow.log_artifacts(str(model_report_dir), artifact_path="evaluation_reports")

                # Log full pipeline model artifact using cloudpickle
                mlflow_sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                    input_example=X_test.iloc[:3]
                )
                print(f"  -> Logged to MLflow Run: 'Regression_{model_name}'")

    # 5. Build and Save Cross-Model Comparison
    leaderboard_df = pd.DataFrame(leaderboard_records).sort_values(by="RMSE_Lakhs", ascending=True)
    leaderboard_csv = REPORTS_REG_DIR / "regression_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False)

    if save_reports:
        fig_comp, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(15, 5))
        melted_df = leaderboard_df.melt(
            id_vars=["Model"],
            value_vars=["RMSE_Lakhs", "MAE_Lakhs"],
            var_name="Metric",
            value_name="Error (Lakhs)"
        )
        sns.barplot(data=melted_df, x="Model", y="Error (Lakhs)", hue="Metric", palette="Set2", ax=ax_c1)
        ax_c1.set_title("Cross-Model Regression Error (Lower is Better)", fontsize=13, fontweight="bold")
        ax_c1.set_xlabel("Model", fontsize=11)
        ax_c1.set_ylabel("Error in ₹ Lakhs", fontsize=11)
        ax_c1.legend(loc="upper right")

        sns.barplot(data=leaderboard_df, x="Model", y="R2_Score", hue="Model", palette="Greens_r", legend=False, ax=ax_c2)
        ax_c2.set_title("R² Goodness of Fit (Higher is Better)", fontsize=13, fontweight="bold")
        ax_c2.set_ylim(0.99, 1.0)
        ax_c2.set_xlabel("Model", fontsize=11)
        ax_c2.set_ylabel("R² Score", fontsize=11)
        for i, model_name in enumerate(leaderboard_df["Model"]):
            r2_val = float(leaderboard_df["R2_Score"].iloc[i])
            ax_c2.text(i, r2_val + 0.0002, f"{r2_val:.4f}", ha="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        comp_img_path = REPORTS_REG_DIR / "regression_comparison.png"
        plt.savefig(comp_img_path, dpi=150)
        plt.close(fig_comp)
        print(f"\n✅ Cross-model comparison saved to: {comp_img_path}")

    print(f"\n{'='*70}")
    print("STEP 3.5 REGRESSION LEADERBOARD SUMMARY")
    print(f"{'='*70}")
    print(leaderboard_df[["Model", "RMSE_Lakhs", "MAE_Lakhs", "R2_Score", "MAPE_Percent", "Inference_Latency_MS"]].to_string(index=False))

    return leaderboard_df


# -----------------------------------------------------------------------------
# Champion Model Registry & Production Staging (Step 3.6)
# -----------------------------------------------------------------------------

def register_champion_models() -> Dict[str, Any]:
    """
    Step 3.6: Champion Model Staging in MLflow Model Registry.

    1. Inspects classification and regression benchmark leaderboards.
    2. Identifies the champion classifier (XGBoost) and champion regressor (LightGBM).
    3. Locates corresponding runs in the 'Real_Estate_Investment_Advisor' experiment.
    4. Registers models in the MLflow Model Registry as:
       - 'Property_Classifier_Prod'
       - 'Property_Regressor_Prod'
    5. Transitions both models to stage 'Production' and assigns aliases ('Production', 'champion').
    6. Attaches descriptive metadata, performance metrics, and operational tags.
    7. Verifies end-to-end model loading via registry URIs.
    """
    print(f"\n{'='*70}")
    print("STEP 3.6: CHAMPION MODEL STAGING IN MLFLOW MODEL REGISTRY")
    print(f"{'='*70}")

    clf_lb_path = REPORTS_CLF_DIR / "classification_leaderboard.csv"
    reg_lb_path = REPORTS_REG_DIR / "regression_leaderboard.csv"

    if not clf_lb_path.exists():
        raise FileNotFoundError(f"Classification leaderboard not found at {clf_lb_path}. Run Step 3.3 first.")
    if not reg_lb_path.exists():
        raise FileNotFoundError(f"Regression leaderboard not found at {reg_lb_path}. Run Step 3.5 first.")

    clf_df = pd.read_csv(clf_lb_path)
    reg_df = pd.read_csv(reg_lb_path)

    # 1. Select Champions
    clf_champ_row = clf_df.iloc[0]
    clf_champ_name = str(clf_champ_row["Model"])

    reg_champ_row = reg_df.iloc[0]
    reg_champ_name = str(reg_champ_row["Model"])

    print(f"🏆 Selected Champion Classifier: '{clf_champ_name}' (F1: {clf_champ_row['F1_Score']:.4f}, Accuracy: {clf_champ_row['Accuracy']*100:.2f}%)")
    print(f"🏆 Selected Champion Regressor : '{reg_champ_name}' (RMSE: ₹{reg_champ_row['RMSE_Lakhs']:.2f}L, R²: {reg_champ_row['R2_Score']:.4f})")

    # 2. Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    exp = client.get_experiment_by_name("Real_Estate_Investment_Advisor")
    if exp is None:
        raise ValueError("MLflow experiment 'Real_Estate_Investment_Advisor' does not exist.")

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=100)
    clf_run_target = f"Classification_{clf_champ_name}"
    reg_run_target = f"Regression_{reg_champ_name}"

    clf_run = next((r for r in runs if r.info.run_name == clf_run_target), None)
    reg_run = next((r for r in runs if r.info.run_name == reg_run_target), None)

    if clf_run is None:
        raise ValueError(f"Could not find MLflow run named '{clf_run_target}'.")
    if reg_run is None:
        raise ValueError(f"Could not find MLflow run named '{reg_run_target}'.")

    # 3. Register Champion Classifier
    print("\n--- Registering Champion Classifier: 'Property_Classifier_Prod' ---")
    clf_model_name = "Property_Classifier_Prod"
    existing_clf_versions = client.search_model_versions(f"name='{clf_model_name}'")
    matching_clf_version = next((v for v in existing_clf_versions if v.run_id == clf_run.info.run_id), None)

    if matching_clf_version is None:
        clf_mv = mlflow.register_model(
            model_uri=f"runs:/{clf_run.info.run_id}/model",
            name=clf_model_name
        )
        clf_version = str(clf_mv.version)
    else:
        clf_version = str(matching_clf_version.version)
        print(f"  -> Found existing version {clf_version} for run {clf_run.info.run_id}")

    client.update_registered_model(
        name=clf_model_name,
        description=(
            "Production Champion Classification Model for Real Estate Investment Advisory. "
            "Predicts whether an Indian residential property is a 'Good Investment' (1) or not (0) "
            "based on property pricing vs. locality medians, physical specs, and civic infrastructure scoring."
        )
    )

    try:
        client.transition_model_version_stage(
            name=clf_model_name,
            version=clf_version,
            stage="Production",
            archive_existing_versions=True
        )
    except Exception as e:
        print(f"  Note: Legacy stage transition: {e}")

    client.set_registered_model_alias(name=clf_model_name, alias="Production", version=clf_version)
    client.set_registered_model_alias(name=clf_model_name, alias="champion", version=clf_version)

    client.update_model_version(
        name=clf_model_name,
        version=clf_version,
        description=(
            f"Algorithm: {clf_champ_name.upper()} | Accuracy: {float(clf_champ_row['Accuracy']):.4f} | "
            f"F1-Score: {float(clf_champ_row['F1_Score']):.4f} | ROC-AUC: {float(clf_champ_row['ROC_AUC']):.4f} | "
            f"Latency: {float(clf_champ_row['Inference_Latency_MS']):.3f} ms | Evaluated on 50,000 test listings."
        )
    )
    client.set_model_version_tag(clf_model_name, clf_version, "task", "classification")
    client.set_model_version_tag(clf_model_name, clf_version, "algorithm", clf_champ_name)
    client.set_model_version_tag(clf_model_name, clf_version, "accuracy", f"{float(clf_champ_row['Accuracy']):.4f}")
    client.set_model_version_tag(clf_model_name, clf_version, "f1_score", f"{float(clf_champ_row['F1_Score']):.4f}")
    client.set_model_version_tag(clf_model_name, clf_version, "roc_auc", f"{float(clf_champ_row['ROC_AUC']):.4f}")
    client.set_model_version_tag(clf_model_name, clf_version, "stage", "Production")
    print(f"  ✅ 'Property_Classifier_Prod' Version {clf_version} staged to 'Production' with alias '@champion'")

    # 4. Register Champion Regressor
    print("\n--- Registering Champion Regressor: 'Property_Regressor_Prod' ---")
    reg_model_name = "Property_Regressor_Prod"
    existing_reg_versions = client.search_model_versions(f"name='{reg_model_name}'")
    matching_reg_version = next((v for v in existing_reg_versions if v.run_id == reg_run.info.run_id), None)

    if matching_reg_version is None:
        reg_mv = mlflow.register_model(
            model_uri=f"runs:/{reg_run.info.run_id}/model",
            name=reg_model_name
        )
        reg_version = str(reg_mv.version)
    else:
        reg_version = str(matching_reg_version.version)
        print(f"  -> Found existing version {reg_version} for run {reg_run.info.run_id}")

    client.update_registered_model(
        name=reg_model_name,
        description=(
            "Production Champion Regression Model for Real Estate Investment Advisory. "
            "Predicts estimated 5-year valuation ('Future_Price_5Y' in ₹ Lakhs) "
            "using localized price medians, physical attributes, and civic infrastructure metrics."
        )
    )

    try:
        client.transition_model_version_stage(
            name=reg_model_name,
            version=reg_version,
            stage="Production",
            archive_existing_versions=True
        )
    except Exception as e:
        print(f"  Note: Legacy stage transition: {e}")

    client.set_registered_model_alias(name=reg_model_name, alias="Production", version=reg_version)
    client.set_registered_model_alias(name=reg_model_name, alias="champion", version=reg_version)

    client.update_model_version(
        name=reg_model_name,
        version=reg_version,
        description=(
            f"Algorithm: {reg_champ_name.upper()} | RMSE: ₹{float(reg_champ_row['RMSE_Lakhs']):.2f}L | "
            f"MAE: ₹{float(reg_champ_row['MAE_Lakhs']):.2f}L | R²: {float(reg_champ_row['R2_Score']):.4f} | "
            f"MAPE: {float(reg_champ_row['MAPE_Percent']):.2f}% | "
            f"Latency: {float(reg_champ_row['Inference_Latency_MS']):.3f} ms | Evaluated on 50,000 test listings."
        )
    )
    client.set_model_version_tag(reg_model_name, reg_version, "task", "regression")
    client.set_model_version_tag(reg_model_name, reg_version, "algorithm", reg_champ_name)
    client.set_model_version_tag(reg_model_name, reg_version, "rmse_lakhs", f"{float(reg_champ_row['RMSE_Lakhs']):.2f}")
    client.set_model_version_tag(reg_model_name, reg_version, "mae_lakhs", f"{float(reg_champ_row['MAE_Lakhs']):.2f}")
    client.set_model_version_tag(reg_model_name, reg_version, "r2_score", f"{float(reg_champ_row['R2_Score']):.4f}")
    client.set_model_version_tag(reg_model_name, reg_version, "stage", "Production")
    print(f"  ✅ 'Property_Regressor_Prod' Version {reg_version} staged to 'Production' with alias '@champion'")

    # 5. Verify Model Loading
    print("\n--- Verifying Production Model Loading ---")
    _ = mlflow.pyfunc.load_model(f"models:/{clf_model_name}/Production")
    _ = mlflow.pyfunc.load_model(f"models:/{reg_model_name}/Production")
    print(f"  ✅ Successfully loaded '{clf_model_name}' via 'models:/{clf_model_name}/Production'")
    print(f"  ✅ Successfully loaded '{reg_model_name}' via 'models:/{reg_model_name}/Production'")

    summary: Dict[str, Any] = {
        "classifier": {
            "registered_name": clf_model_name,
            "version": clf_version,
            "algorithm": clf_champ_name,
            "stage": "Production",
            "aliases": ["Production", "champion"],
            "run_id": clf_run.info.run_id,
            "accuracy": float(clf_champ_row["Accuracy"]),
            "f1_score": float(clf_champ_row["F1_Score"]),
            "roc_auc": float(clf_champ_row["ROC_AUC"]),
            "latency_ms": float(clf_champ_row["Inference_Latency_MS"])
        },
        "regressor": {
            "registered_name": reg_model_name,
            "version": reg_version,
            "algorithm": reg_champ_name,
            "stage": "Production",
            "aliases": ["Production", "champion"],
            "run_id": reg_run.info.run_id,
            "rmse_lakhs": float(reg_champ_row["RMSE_Lakhs"]),
            "mae_lakhs": float(reg_champ_row["MAE_Lakhs"]),
            "r2_score": float(reg_champ_row["R2_Score"]),
            "mape_percent": float(reg_champ_row["MAPE_Percent"]),
            "latency_ms": float(reg_champ_row["Inference_Latency_MS"])
        }
    }

    print(f"\n{'='*70}")
    print("STEP 3.6 COMPLETE: ALL CHAMPION MODELS STAGED IN MLFLOW REGISTRY")
    print(f"{'='*70}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real Estate Advisory ML Pipeline")
    parser.add_argument("--step", type=str, default="3.6", help="Pipeline step to run: 3.2 (train clf), 3.3 (eval clf), 3.4 (train reg), 3.5 (eval reg), 3.6 (register champions)")
    args = parser.parse_args()

    if args.step == "3.2":
        train_classification_models()
    elif args.step == "3.3":
        evaluate_classification_models()
    elif args.step == "3.4":
        train_regression_models()
    elif args.step == "3.5":
        evaluate_regression_models()
    elif args.step in ["3.6", "register"]:
        register_champion_models()
    else:
        print(f"Unknown step: {args.step}. Running Step 3.6 register champions by default.")
        register_champion_models()
