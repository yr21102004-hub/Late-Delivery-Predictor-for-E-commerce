"""
src/model_evaluation.py
-----------------------
Model evaluation metrics, diagnostic curves, comparison leaderboards, and feature importance.

Responsibility:
    - compute_metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
    - ModelEvaluator: Diagnostic plots (Confusion Matrix, ROC Curve, PR Curve, Learning Curves).
    - ModelComparator: Multi-model evaluation benchmark, comparative bar charts, and feature importance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.config import CFG, get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "model",
    split: str = "unknown",
) -> Dict[str, Any]:
    """Compute complete classification metrics suite."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    cr   = classification_report(y_true, y_pred, target_names=["On Time", "Late"])

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = None

    res = {
        "model_name":            model_name,
        "split":                 split,
        "accuracy":              round(acc, 4),
        "precision":             round(prec, 4),
        "recall":                round(rec, 4),
        "f1_score":              round(f1, 4),
        "roc_auc":               round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix":      cm.tolist(),
        "classification_report": cr,
    }
    logger.info(
        "[%s | %s] Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  ROC-AUC=%s",
        model_name, split, acc, prec, rec, f1,
        f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
    )
    return res


def save_metrics_json(metrics: Dict[str, Any], filename: str) -> Path:
    path = CFG.metrics_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return path


class ModelEvaluator:
    """Computes evaluation metrics and generates diagnostic curves."""

    def evaluate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "model",
        split: str = "dev",
    ) -> Dict[str, Any]:
        y_pred = pipeline.predict(X)
        y_prob = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None
        return compute_metrics(np.array(y), y_pred, y_prob, model_name=model_name, split=split)

    def plot_confusion_matrix(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str, split: str) -> Path:
        y_pred = pipeline.predict(X)
        cm = confusion_matrix(y, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["On Time", "Late"], yticklabels=["On Time", "Late"], ax=axes[0])
        axes[0].set_title(f"{model_name} — Confusion Matrix ({split})\nRaw Counts")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=["On Time", "Late"], yticklabels=["On Time", "Late"], ax=axes[1])
        axes[1].set_title(f"{model_name} — Confusion Matrix ({split})\nNormalized")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        plt.tight_layout()
        fname = f"confusion_matrix_{model_name.replace(' ', '_').lower()}_{split}.png"
        path = CFG.figure_path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_roc_curve(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str, split: str) -> Optional[Path]:
        if not hasattr(pipeline, "predict_proba"):
            return None
        y_prob = pipeline.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, color="#2563EB", label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Baseline (AUC = 0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name} ({split})")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"roc_curve_{model_name.replace(' ', '_').lower()}_{split}.png"
        path = CFG.figure_path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_precision_recall_curve(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str, split: str) -> Optional[Path]:
        if not hasattr(pipeline, "predict_proba"):
            return None
        y_prob = pipeline.predict_proba(X)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall, precision)
        baseline = y.mean()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, lw=2, color="#DC2626", label=f"PR curve (AUC = {pr_auc:.4f})")
        ax.axhline(y=baseline, color="k", linestyle="--", lw=1.5, label=f"Baseline Prevalence ({baseline:.1%})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve — {model_name} ({split})")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"pr_curve_{model_name.replace(' ', '_').lower()}_{split}.png"
        path = CFG.figure_path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_learning_curve(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str, cv: int = 3) -> Path:
        from sklearn.model_selection import learning_curve
        train_sizes = [0.2, 0.5, 0.8, 1.0]
        train_sizes_abs, train_scores, val_scores = learning_curve(
            pipeline, X, y, train_sizes=train_sizes, cv=cv, scoring="f1", n_jobs=-1, random_state=CFG.random_state
        )
        train_mean = train_scores.mean(axis=1)
        val_mean   = val_scores.mean(axis=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_sizes_abs, train_mean, "o-", color="#2563EB", label="Training F1")
        ax.plot(train_sizes_abs, val_mean, "s-", color="#DC2626", label="Cross-Validation F1")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Learning Curve — {model_name}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fname = f"learning_curve_{model_name.replace(' ', '_').lower()}.png"
        path = CFG.figure_path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path


class ModelComparator:
    """Benchmarks multiple fitted models, builds comparison tables, and plots leaderboard charts."""

    def __init__(self) -> None:
        self.evaluator = ModelEvaluator()

    def compare(
        self,
        fitted_pipelines: Dict[str, Pipeline],
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
    ) -> Tuple[pd.DataFrame, str, Pipeline]:
        rows = []
        for name, pipeline in fitted_pipelines.items():
            metrics = self.evaluator.evaluate(pipeline, X_dev, y_dev, model_name=name, split="dev")
            row = {k: v for k, v in metrics.items() if k not in ("confusion_matrix", "classification_report")}
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
        csv_path = CFG.reports_path("model_comparison.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Saved model comparison table → %s", csv_path)

        best_name = df.iloc[0]["model_name"]
        best_pipeline = fitted_pipelines[best_name]

        self.plot_comparison_bar(df)
        return df, best_name, best_pipeline

    def plot_comparison_bar(self, df: pd.DataFrame) -> Path:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        metrics_present = [m for m in metrics_to_plot if m in df.columns]

        n_models  = len(df)
        n_metrics = len(metrics_present)
        x = np.arange(n_models)
        width = 0.15

        fig, ax = plt.subplots(figsize=(max(10, n_models * 1.8), 5.5))
        colors = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]

        for i, metric in enumerate(metrics_present):
            vals = df[metric].fillna(0).values
            bars = ax.bar(x + i * width, vals, width, label=metric.upper(), color=colors[i % len(colors)], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)

        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(df["model_name"], rotation=20, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Model Leaderboard — Holdout Development Set")
        ax.legend(loc="upper right", ncol=3)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = CFG.figure_path("model_comparison_bar.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_feature_importance(self, pipeline: Pipeline, model_name: str, top_n: int = 15) -> Optional[Path]:
        clf = pipeline.named_steps.get("classifier")
        if clf is None:
            return None

        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

        importances = None
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])

        if importances is None:
            return None

        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"feat_{i}" for i in range(len(importances))]

        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_vals = importances[indices]

        fig, ax = plt.subplots(figsize=(8, max(4.5, top_n * 0.3)))
        ax.barh(range(len(top_features)), top_vals[::-1], color=plt.cm.viridis(np.linspace(0.4, 0.9, top_n))[::-1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_n} Predictive Features — {model_name}")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fname = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
        path = CFG.figure_path(fname)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

