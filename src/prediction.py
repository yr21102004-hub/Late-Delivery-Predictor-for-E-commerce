"""
src/prediction.py
-----------------
Production inference engine for real-time and batch late delivery predictions.

Responsibility:
    - Loads serialized Pickle pipeline.
    - Validates order inputs.
    - Returns predicted binary class, probability, risk tiers (Low/Medium/High), and operational messages.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import CFG, get_logger

logger = get_logger(__name__)

_LOW_THRESHOLD = 0.30
_MEDIUM_THRESHOLD = 0.60


class Predictor:
    """Production inference interface loading the pre-trained Scikit-Learn pipeline."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or CFG.model_path()
        self.pipeline = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run train.py first.")
        with open(self.model_path, "rb") as fh:
            pipeline = pickle.load(fh)
        logger.info("Loaded trained model from %s", self.model_path)
        return pipeline

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < _LOW_THRESHOLD:
            return "LOW"
        elif prob < _MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _business_message(predicted_class: int, prob: float, risk: str) -> str:
        if predicted_class == 0:
            return f"Order predicted to arrive ON TIME ({prob:.1%} delay risk). Standard dispatch."
        return (
            f"[WARNING] Order has a {prob:.1%} probability of arriving LATE (Risk: {risk}). "
            "Consider priority packing or carrier upgrade."
        )

    def _validate_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        expected = CFG.numerical_features + CFG.categorical_features
        missing = [c for c in expected if c not in data]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")

        df = pd.DataFrame([data])
        for col in CFG.numerical_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in CFG.categorical_features:
            df[col] = df[col].astype(str)
        return df[expected]

    def predict_single(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Predict delay probability for a single order payload."""
        X = self._validate_input(order)
        pred_class = int(self.pipeline.predict(X)[0])
        prob = float(self.pipeline.predict_proba(X)[0][1])
        risk = self._risk_level(prob)
        msg = self._business_message(pred_class, prob, risk)

        return {
            "predicted_class":  pred_class,
            "late_probability": prob,
            "risk_level":       risk,
            "message":          msg,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of orders from a DataFrame."""
        expected = CFG.numerical_features + CFG.categorical_features
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Input batch DataFrame missing required columns: {missing}")

        X = df[expected].copy()
        for col in CFG.numerical_features:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        for col in CFG.categorical_features:
            X[col] = X[col].astype(str)

        preds = self.pipeline.predict(X)
        probs = self.pipeline.predict_proba(X)[:, 1]
        risks = [self._risk_level(p) for p in probs]

        res = df.copy()
        res["predicted_class"]  = preds
        res["late_probability"] = probs
        res["risk_level"]       = risks
        return res

