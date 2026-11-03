"""
src/data_processing.py
----------------------
Consolidated data extraction, cleaning, merging, and feature engineering module.

Responsibility:
    - DataLoader: Read raw CSV tables from data/raw/.
    - DataCleaner: Clean, filter, validate integrity, and create binary target variable.
    - DataMerger: Aggregate multi-row tables, join relational data, and compute vectorized Haversine distance.
    - FeatureEngineer: Extract temporal, volumetric, and ratio features strictly available at order placement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import CFG, get_logger

logger = get_logger(__name__)

# Date columns in orders table
_ORDER_DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def haversine_vectorized(
    lat1: np.ndarray, lng1: np.ndarray,
    lat2: np.ndarray, lng2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized great-circle distance (km) between coordinates using Haversine formula.
    Accounts for Earth's curvature across continental Brazil (~4,300 km span).
    """
    R = 6371.0  # Earth's radius in kilometers

    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat   = np.radians(lat2 - lat1)
    dlng   = np.radians(lng2 - lng1)

    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlng / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


class DataLoader:
    """Encapsulates file loading for all raw Olist CSV tables."""

    def load_table(self, table_key: str) -> pd.DataFrame:
        path = CFG.raw_data_path(table_key)
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows × %d cols", table_key, *df.shape)
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading all Olist relational tables...")
        return {
            "orders":               self.load_table("orders"),
            "customers":            self.load_table("customers"),
            "order_items":          self.load_table("order_items"),
            "payments":             self.load_table("payments"),
            "products":             self.load_table("products"),
            "sellers":              self.load_table("sellers"),
            "geolocation":          self.load_table("geolocation"),
            "reviews":              self.load_table("reviews"),
            "category_translation": self.load_table("category_translation"),
        }


class DataCleaner:
    """Cleans raw Olist tables, validates integrity, and creates the target variable."""

    def clean_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in _ORDER_DATE_COLS:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Keep only delivered orders where target can be reliably derived
        df = df[df["order_status"] == "delivered"].copy()
        df = df.dropna(subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "order_purchase_timestamp",
        ])

        # Target variable: 1 = Late, 0 = On Time
        df["Late_Delivery"] = (
            df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
        ).astype(int)

        # Integrity checks: purge logically invalid timestamps
        valid_dates = (
            (df["order_delivered_customer_date"] >= df["order_purchase_timestamp"]) &
            (df["order_estimated_delivery_date"] >= df["order_purchase_timestamp"])
        )
        df = df[valid_dates].copy()
        logger.info("Cleaned orders: %d rows (Late rate: %.2f%%)", len(df), 100 * df["Late_Delivery"].mean())
        return df

    def clean_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["customer_id"]).copy()
        df["customer_state"] = df["customer_state"].str.upper().str.strip()
        return df

    def clean_order_items(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates().copy()
        df = df[(df["price"] > 0) & (df["freight_value"] >= 0)].copy()
        return df

    def clean_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["payment_type"] != "not_defined"].drop_duplicates().copy()
        return df

    def clean_products(self, df: pd.DataFrame, translation_df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["product_id"]).copy()
        df = df.merge(translation_df, on="product_category_name", how="left")
        df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")

        # Impute missing physical dimensions with medians
        for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
            df[col] = df[col].fillna(df[col].median())
        return df

    def clean_sellers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["seller_id"]).copy()
        df["seller_state"] = df["seller_state"].str.upper().str.strip()
        return df

    def clean_geolocation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates().copy()
        # Filter coordinates to Brazil bounding box
        valid_coords = (
            df["geolocation_lat"].between(-33.75, 5.27) &
            df["geolocation_lng"].between(-73.99, -34.72)
        )
        df = df[valid_coords]
        # Average per postal code prefix to reduce 1M records to unique spatial coordinates
        return (
            df.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
            .mean()
            .reset_index()
        )

    def clean_all(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        logger.info("Executing full cleaning pipeline across all tables...")
        return {
            "orders":      self.clean_orders(tables["orders"]),
            "customers":   self.clean_customers(tables["customers"]),
            "order_items": self.clean_order_items(tables["order_items"]),
            "payments":    self.clean_payments(tables["payments"]),
            "products":    self.clean_products(tables["products"], tables["category_translation"]),
            "sellers":     self.clean_sellers(tables["sellers"]),
            "geolocation": self.clean_geolocation(tables["geolocation"]),
        }


class DataMerger:
    """Aggregates sub-tables and merges relational entities into a unified order-level DataFrame."""

    def _aggregate_items(self, df: pd.DataFrame) -> pd.DataFrame:
        first_item = (
            df.sort_values("order_item_id")
            .groupby("order_id")
            .first()[["product_id", "seller_id"]]
            .reset_index()
        )
        agg = df.groupby("order_id").agg(
            n_items       =("order_item_id", "max"),
            n_sellers     =("seller_id", "nunique"),
            price         =("price", "sum"),
            freight_value =("freight_value", "sum"),
        ).reset_index()
        return agg.merge(first_item, on="order_id", how="left")

    def _aggregate_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        first_pay = (
            df.sort_values("payment_sequential")
            .groupby("order_id")
            .first()[["payment_type"]]
            .reset_index()
        )
        agg = df.groupby("order_id").agg(
            payment_installments=("payment_installments", "max"),
            total_payment_value =("payment_value", "sum"),
        ).reset_index()
        agg = agg.merge(first_pay, on="order_id", how="left")
        agg["payment_type"] = agg["payment_type"].fillna("unknown")
        return agg

    def merge_all(self, cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("Merging cleaned tables...")
        orders      = cleaned["orders"]
        customers   = cleaned["customers"]
        items_agg   = self._aggregate_items(cleaned["order_items"])
        pay_agg     = self._aggregate_payments(cleaned["payments"])
        products    = cleaned["products"]
        sellers     = cleaned["sellers"]
        geo         = cleaned["geolocation"]

        seller_geo = geo.rename(columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng",
        })
        cust_geo = geo.rename(columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng",
        })

        # Relational joins
        df = orders.merge(customers[["customer_id", "customer_state", "customer_zip_code_prefix"]], on="customer_id", how="left")
        df = df.merge(items_agg, on="order_id", how="left")
        df = df.merge(pay_agg, on="order_id", how="left")
        product_cols = ["product_id", "product_category_name_english", "product_weight_g",
                        "product_length_cm", "product_height_cm", "product_width_cm"]
        df = df.merge(products[product_cols], on="product_id", how="left")
        df = df.merge(sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]], on="seller_id", how="left")
        df = df.merge(seller_geo, on="seller_zip_code_prefix", how="left")
        df = df.merge(cust_geo, on="customer_zip_code_prefix", how="left")

        # Vectorized Haversine distance
        df["geo_distance_km"] = haversine_vectorized(
            df["seller_lat"].values, df["seller_lng"].values,
            df["customer_lat"].values, df["customer_lng"].values,
        )
        df["geo_distance_km"] = df["geo_distance_km"].fillna(df["geo_distance_km"].median())

        logger.info("Merged dataset ready: %d rows × %d cols", *df.shape)
        return df

    def save(self, df: pd.DataFrame, file_key: str = "merged_dataset") -> None:
        path = CFG.processed_data_path(file_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved merged data → %s", path)


class FeatureEngineer:
    """Transforms raw order data into leakage-free predictive features."""

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ts = pd.to_datetime(df["order_purchase_timestamp"])
        est_date = pd.to_datetime(df["order_estimated_delivery_date"])

        # Temporal features
        df["purchase_month"]       = ts.dt.month
        df["purchase_day_of_week"] = ts.dt.dayofweek
        df["purchase_hour"]        = ts.dt.hour

        # Promised delivery duration (days)
        df["estimated_delivery_days"] = (est_date - ts).dt.days.clip(lower=0)

        # Volumetric & cost ratio features
        df["product_volume_cm3"] = (
            df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
        )
        df["freight_to_price_ratio"] = (
            df["freight_value"] / df["price"].replace(0, np.nan)
        ).fillna(0.0)

        df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")
        return df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to strictly model-ready features + target column."""
        keep = CFG.numerical_features + CFG.categorical_features + [CFG.target_column]
        return df[keep].copy()

    def save(self, df: pd.DataFrame, file_key: str = "featured_dataset") -> None:
        path = CFG.processed_data_path(file_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved featured dataset → %s", path)

