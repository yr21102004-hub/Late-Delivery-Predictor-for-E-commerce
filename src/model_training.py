"""
src/model_training.py
---------------------
Model initialization, stratified data splitting, model training, and hyperparameter tuning.

Responsibility:
    - DataSplitter: Stratified 70% Train, 15% Dev, 15% Test partition.
    - Model factories: Pipeline wrappers combining preprocessor and classifiers.
    - ModelTrainer: Training, cross-validation, and RandomizedSearchCV tuning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import CFG, get_logger
from src.preprocessing import build_preprocessor

logger = get_logger(__name__)


class DataSplitter:
    """Executes stratified Train (70%), Dev (15%), and Test (15%) dataset partition."""

    def split(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        target_col = target_col or CFG.target_column
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 1. Carve out holdout Test set (15%)
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=CFG.test_size, random_state=CFG.random_state)
        train_dev_idx, test_idx = next(sss_test.split(X, y))
        X_train_dev, X_test = X.iloc[train_dev_idx], X.iloc[test_idx]
        y_train_dev, y_test = y.iloc[train_dev_idx], y.iloc[test_idx]

        # 2. Split remaining 85% into Train (70%) and Dev (15%)
        dev_fraction = CFG.dev_size / (CFG.train_size + CFG.dev_size)
        sss_dev = StratifiedShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=CFG.random_state)
        train_idx, dev_idx = next(sss_dev.split(X_train_dev, y_train_dev))
        X_train, X_dev = X_train_dev.iloc[train_idx], X_train_dev.iloc[dev_idx]
        y_train, y_dev = y_train_dev.iloc[train_idx], y_train_dev.iloc[dev_idx]

        # Save partitions
        proc = CFG.proc_data_dir
        proc.mkdir(parents=True, exist_ok=True)
        for name, sub_X, sub_y in [("train", X_train, y_train), ("dev", X_dev, y_dev), ("test", X_test, y_test)]:
            out = sub_X.copy()
            out[CFG.target_column] = sub_y.values
            out.to_csv(proc / f"{name}_set.csv", index=False)
            logger.info("Saved %s set: %d rows (Late rate: %.1f%%)", name, len(sub_y), 100 * sub_y.mean())

        return X_train, X_dev, X_test, y_train, y_dev, y_test


def _build_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier",   classifier),
    ])


def get_baseline_model() -> Pipeline:
    return _build_pipeline(DummyClassifier(strategy="most_frequent", random_state=CFG.random_state))


def get_logistic_regression() -> Pipeline:
    return _build_pipeline(LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=CFG.random_state
    ))


def get_knn() -> Pipeline:
    return _build_pipeline(KNeighborsClassifier(n_neighbors=15, metric="euclidean", n_jobs=-1))


def get_decision_tree() -> Pipeline:
    return _build_pipeline(DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=CFG.random_state
    ))


def get_random_forest() -> Pipeline:
    return _build_pipeline(RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=10,
        class_weight="balanced_subsample", n_jobs=-1, random_state=CFG.random_state
    ))


def get_gradient_boosting() -> Pipeline:
    return _build_pipeline(GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.08, max_depth=4,
        min_samples_leaf=20, subsample=0.8, random_state=CFG.random_state
    ))


class ModelTrainer:
    """Manages baseline initialization, candidate fitting, cross-validation, and hyperparameter tuning."""

    def __init__(self) -> None:
        self.model_registry: Dict[str, Pipeline] = {
            "Baseline (Dummy)":    get_baseline_model(),
            "Logistic Regression": get_logistic_regression(),
            "KNN":                 get_knn(),
            "Decision Tree":       get_decision_tree(),
            "Random Forest":       get_random_forest(),
            "Gradient Boosting":   get_gradient_boosting(),
        }

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
        fitted = {}
        for name, pipeline in self.model_registry.items():
            logger.info("Training %s...", name)
            pipeline.fit(X_train, y_train)
            fitted[name] = pipeline
        return fitted

    def cross_validate_model(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scoring: str = "f1",
        cv: int = 3,
    ) -> Dict[str, float]:
        clf = pipeline.named_steps.get("classifier")
        # For non-parametric KNN, use a stratified 10k subsample for fast distance computation
        if isinstance(clf, KNeighborsClassifier) and len(X_train) > 10000:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=CFG.random_state)
            idx, _ = next(sss.split(X_train, y_train))
            X_cv, y_cv = X_train.iloc[idx], y_train.iloc[idx]
        else:
            X_cv, y_cv = X_train, y_train

        scores = cross_val_score(pipeline, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=-1)
        res = {"mean_cv_score": float(scores.mean()), "std_cv_score": float(scores.std())}
        logger.info("CV (%d-fold, %s): %.4f ± %.4f", cv, scoring, res["mean_cv_score"], res["std_cv_score"])
        return res

    def tune_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 15) -> Pipeline:
        param_dist = {
            "classifier__n_estimators":     [80, 120],
            "classifier__max_depth":        [8, 12, 16],
            "classifier__min_samples_leaf": [10, 20],
            "classifier__max_features":     ["sqrt", 0.5],
        }
        search = RandomizedSearchCV(
            get_random_forest(), param_distributions=param_dist, n_iter=n_iter,
            scoring="f1", cv=3, n_jobs=-1, random_state=CFG.random_state, refit=True
        )
        logger.info("Running RandomizedSearchCV on Random Forest...")
        search.fit(X_train, y_train)
        logger.info("Best RF params: %s | CV F1: %.4f", search.best_params_, search.best_score_)
        return search.best_estimator_

    def tune_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 10) -> Pipeline:
        param_dist = {
            "classifier__n_estimators":     [80, 100],
            "classifier__learning_rate":    [0.05, 0.1],
            "classifier__max_depth":        [3, 4],
            "classifier__subsample":        [0.8, 1.0],
            "classifier__min_samples_leaf": [15, 25],
        }
        search = RandomizedSearchCV(
            get_gradient_boosting(), param_distributions=param_dist, n_iter=n_iter,
            scoring="f1", cv=3, n_jobs=-1, random_state=CFG.random_state, refit=True
        )
        logger.info("Running RandomizedSearchCV on Gradient Boosting...")
        search.fit(X_train, y_train)
        logger.info("Best GB params: %s | CV F1: %.4f", search.best_params_, search.best_score_)
        return search.best_estimator_

    def refit_on_train_dev(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_dev: pd.DataFrame,
        y_train: pd.Series,
        y_dev: pd.Series,
    ) -> Pipeline:
        X_combined = pd.concat([X_train, X_dev], axis=0)
        y_combined = pd.concat([y_train, y_dev], axis=0)
        logger.info("Refitting selected pipeline on Train + Dev (%d records)...", len(X_combined))
        pipeline.fit(X_combined, y_combined)
        return pipeline

