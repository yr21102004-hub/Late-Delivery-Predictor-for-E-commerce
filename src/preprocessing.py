"""
src/preprocessing.py
--------------------
Scikit-Learn preprocessing pipeline for heterogeneous numerical and categorical features.

Responsibility:
    - Builds a reusable ColumnTransformer.
    - Numerical: SimpleImputer (median) -> StandardScaler.
    - Categorical: SimpleImputer (constant='unknown') -> OneHotEncoder(handle_unknown='ignore').
"""

from __future__ import annotations

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CFG, get_logger

logger = get_logger(__name__)


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct the Scikit-learn ColumnTransformer pipeline.

    Parameters
    ----------
    numerical_features : list of str, optional
    categorical_features : list of str, optional

    Returns
    -------
    ColumnTransformer
    """
    num_features = numerical_features or CFG.numerical_features
    cat_features = categorical_features or CFG.categorical_features

    logger.debug(
        "Building preprocessor pipeline: %d numerical, %d categorical features.",
        len(num_features), len(cat_features),
    )

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numerical",   numerical_pipeline,   num_features),
            ("categorical", categorical_pipeline, cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

