import time
import json
import pandas as pd
import optuna
import dagshub
from dotenv import load_dotenv
from pathlib import Path
import joblib
import seaborn as sns
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
load_dotenv()

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine
from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.utils.exception import CustomException
from CustomerChurnPrediction.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    """Evaluates the base and tuned models on a held-out test set, logs results to MLflow, and saves scores to JSON."""

    THRESHOLD = 0.25

    def __init__(self, config: ModelEvaluationConfig):
        """Sets up config and connects MLflow tracking to DagsHub."""
        self.config = config

        dagshub.init(
            repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
            repo_name=os.getenv("DAGSHUB_REPO_NAME"),
            mlflow=True,
        )

    def get_input_data(self) -> pd.DataFrame:
        """Loads the transformed data used for training."""
        logger.info(f"Loading data from: {self.config.input_data_path}")
        return pd.read_csv(self.config.input_data_path)

    def split_data(self, df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2):
        """Reproduces the same train/test split used during training (same seed)."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def load_models(self):
        """Loads the saved base and tuned models from disk."""
        base_model = joblib.load(self.config.base_model_path)
        tuned_model = joblib.load(self.config.tuned_model_path)
        logger.info("Loaded base and tuned models")
        return base_model, tuned_model

    def evaluate_model(self, model, X_test, y_test):
        """Computes precision, recall, f1, and roc_auc for a model at the fixed threshold."""
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= self.THRESHOLD).astype(int)

        metrics = {
            "precision": precision_score(y_test, y_pred, pos_label=1),
            "recall": recall_score(y_test, y_pred, pos_label=1),
            "f1": f1_score(y_test, y_pred, pos_label=1),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        logger.info(f"Metrics: {metrics}")
        logger.info("\n" + classification_report(y_test, y_pred, digits=3))

        return metrics, y_pred

    def log_confusion_matrix(self, title: str, save_path: str, y_test, y_pred) -> str:
        """Builds a confusion matrix plot for a model and saves it as a PNG."""
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {title}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Confusion matrix saved to {save_path}")
        return save_path
    
    def plot_roc_curve(self, base_model, tuned_model, X_test, y_test) -> None:
        """Saves an ROC curve plot comparing the base and tuned models."""
        fig, ax = plt.subplots(figsize=(7, 6))

        for name, model in [("Base XGBoost", base_model), ("Tuned XGBoost", tuned_model)]:
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve -- Base vs Tuned XGBoost")
        ax.legend(loc="lower right")
        ax.grid(True)

        Path(self.config.roc_curve_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.config.roc_curve_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC curve saved to {self.config.roc_curve_path}")

    def log_to_mlflow(self, run_name: str, model, metrics: dict, cm_path: str,roc_path: str) -> None:
        """Logs a model's hyperparameters, evaluation metrics, and confusion matrix to MLflow."""
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(model.get_params())
            mlflow.log_param("threshold", self.THRESHOLD)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)

        logger.info(f"Logged '{run_name}' evaluation run to MLflow")

    def save_scores(self, scores: dict) -> None:
        """Saves all models' metrics to a single JSON file."""
        os.makedirs(os.path.dirname(self.config.scores_file_path), exist_ok=True)

        with open(self.config.scores_file_path, "w") as f:
            json.dump(scores, f, indent=4)

        logger.info(f"Scores saved to {self.config.scores_file_path}")

    def run_evaluation(self) -> None:
        """Runs the full evaluation pipeline: load data/models, evaluate both, log to MLflow, save scores.json."""
        df = self.get_input_data()
        _, X_test, _, y_test = self.split_data(df)

        base_model, tuned_model = self.load_models()

        base_metrics, base_pred = self.evaluate_model(base_model, X_test, y_test)
        tuned_metrics, tuned_pred = self.evaluate_model(tuned_model, X_test, y_test)

        base_cm_path = self.log_confusion_matrix(
            "XGBoost", self.config.base_cm_path, y_test, base_pred
        )
        tuned_cm_path = self.log_confusion_matrix(
            "XGBoost_Tuned", self.config.tuned_cm_path, y_test, tuned_pred
        )
        self.plot_roc_curve(base_model, tuned_model, X_test, y_test)
        # self.log_to_mlflow("XGBoost_Eval", base_model, base_metrics, base_cm_path)
        # self.log_to_mlflow("XGBoost_Tuned_Eval", tuned_model, tuned_metrics, tuned_cm_path)

        scores = {
            "base_model": base_metrics,
            "tuned_model": tuned_metrics,
        }
        self.save_scores(scores)
