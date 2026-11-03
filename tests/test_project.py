"""
tests/test_project.py
---------------------
Consolidated test suite covering data cleaning, preprocessing, and inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CFG
from src.data_processing import DataCleaner
from src.prediction import Predictor
from src.preprocessing import build_preprocessor


# --------------------------------------------------------------------------- #
# Data Cleaning Tests                                                         #
# --------------------------------------------------------------------------- #

def test_clean_orders_target_and_filtering():
    df = pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4"],
        "customer_id": ["c1", "c2", "c3", "c4"],
        "order_status": ["delivered", "delivered", "canceled", "delivered"],
        "order_purchase_timestamp": [
            "2018-01-01 10:00:00",
            "2018-01-02 12:00:00",
            "2018-01-03 08:00:00",
            "2018-01-04 14:00:00",
        ],
        "order_approved_at": [
            "2018-01-01 11:00:00",
            "2018-01-02 13:00:00",
            None,
            "2018-01-04 15:00:00",
        ],
        "order_delivered_carrier_date": [
            "2018-01-02 10:00:00",
            "2018-01-03 10:00:00",
            None,
            "2018-01-05 10:00:00",
        ],
        "order_delivered_customer_date": [
            "2018-01-05 15:00:00",  # On time (est: 2018-01-10) -> 0
            "2018-01-15 15:00:00",  # Late (est: 2018-01-10) -> 1
            None,
            "2018-01-10 10:00:00",  # Late (delivery time > 00:00 est) -> 1
        ],
        "order_estimated_delivery_date": [
            "2018-01-10 00:00:00",
            "2018-01-10 00:00:00",
            "2018-01-12 00:00:00",
            "2018-01-10 00:00:00",
        ],
    })

    cleaner = DataCleaner()
    cleaned = cleaner.clean_orders(df)

    assert len(cleaned) == 3
    assert (cleaned["order_status"] == "delivered").all()
    assert "Late_Delivery" in cleaned.columns

    labels = cleaned.set_index("order_id")["Late_Delivery"].to_dict()
    assert labels["o1"] == 0
    assert labels["o2"] == 1
    assert labels["o4"] == 1


def test_clean_customers():
    cleaner = DataCleaner()
    df = pd.DataFrame({
        "customer_id": ["c1", "c1", "c2"],
        "customer_unique_id": ["u1", "u1", "u2"],
        "customer_zip_code_prefix": [12345, 12345, 67890],
        "customer_city": ["sao paulo", "sao paulo", "rio de janeiro"],
        "customer_state": [" sp ", "sp", "rj"],
    })
    res = cleaner.clean_customers(df)
    assert len(res) == 2
    assert list(res["customer_state"]) == ["SP", "RJ"]


# --------------------------------------------------------------------------- #
# Preprocessing Pipeline Tests                                                #
# --------------------------------------------------------------------------- #

def test_build_preprocessor():
    num_cols = ["feat_num1", "feat_num2"]
    cat_cols = ["feat_cat1"]

    preprocessor = build_preprocessor(
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )

    df = pd.DataFrame({
        "feat_num1": [10.0, np.nan, 30.0],
        "feat_num2": [100.0, 200.0, 300.0],
        "feat_cat1": ["A", "B", np.nan],
    })

    X_trans = preprocessor.fit_transform(df)
    assert X_trans is not None
    assert X_trans.shape[0] == 3
    assert not np.isnan(X_trans).any()


def test_unseen_category_handling():
    num_cols = ["num"]
    cat_cols = ["cat"]

    preprocessor = build_preprocessor(
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )

    df_train = pd.DataFrame({
        "num": [1.0, 2.0],
        "cat": ["electronics", "apparel"],
    })
    preprocessor.fit(df_train)

    df_test = pd.DataFrame({
        "num": [3.0],
        "cat": ["brand_new_unseen_category"],
    })
    X_test_trans = preprocessor.transform(df_test)
    assert X_test_trans is not None
    assert not np.isnan(X_test_trans).any()


# --------------------------------------------------------------------------- #
# Predictor & Inference Tests                                                 #
# --------------------------------------------------------------------------- #

SAMPLE_PAYLOAD = {
    "purchase_month": 5,
    "purchase_day_of_week": 2,
    "purchase_hour": 14,
    "estimated_delivery_days": 20,
    "product_weight_g": 500.0,
    "product_volume_cm3": 4000.0,
    "freight_value": 20.0,
    "price": 100.0,
    "freight_to_price_ratio": 0.2,
    "n_items": 1,
    "n_sellers": 1,
    "payment_installments": 2,
    "total_payment_value": 120.0,
    "geo_distance_km": 350.0,
    "product_category_name_english": "telephony",
    "seller_state": "SP",
    "customer_state": "SP",
    "payment_type": "credit_card",
}


def test_predict_single_and_validation():
    if CFG.model_path().exists():
        predictor = Predictor()

        # Missing input validation
        with pytest.raises(ValueError, match="Missing required input fields"):
            predictor.predict_single({"purchase_month": 5})

        # Correct inference
        res = predictor.predict_single(SAMPLE_PAYLOAD)
        assert "predicted_class" in res
        assert res["predicted_class"] in [0, 1]
        assert 0.0 <= res["late_probability"] <= 1.0
        assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(res["message"], str)


def test_compute_metrics():
    from src.model_evaluation import compute_metrics
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1])

    res = compute_metrics(y_true, y_pred, y_prob, model_name="test_clf", split="test")
    assert res["accuracy"] == 0.8
    assert res["precision"] == 1.0
    assert res["recall"] == 0.5
    assert "confusion_matrix" in res
    assert "classification_report" in res

