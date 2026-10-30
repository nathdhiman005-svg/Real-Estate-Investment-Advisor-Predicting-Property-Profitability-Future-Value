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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
)

# Silence MLflow agent telemetry hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
import mlflow.sklearn as mlflow_sklearn

from src.preprocessing import get_train_test_split, build_pipeline, get_feature_names
from src.utils import MODELS_CLF_DIR, REPORTS_CLF_DIR, MLFLOW_TRACKING_URI

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real Estate Advisory ML Pipeline")
    parser.add_argument("--step", type=str, default="3.3", help="Pipeline step to run: 3.2 (train clf), 3.3 (eval clf)")
    args = parser.parse_args()

    if args.step == "3.2":
        train_classification_models()
    elif args.step == "3.3":
        evaluate_classification_models()
    else:
        print(f"Unknown step: {args.step}. Running Step 3.3 evaluation by default.")
        evaluate_classification_models()
