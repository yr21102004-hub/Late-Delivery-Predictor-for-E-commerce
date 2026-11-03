"""
src/visualization.py
--------------------
Generates all publication-quality Exploratory Data Analysis (EDA) charts.

Responsibility:
    - Univariate, bivariate, multivariate, and correlation visualizations.
    - Saves all generated figures into reports/figures/.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CFG, get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})
sns.set_theme(style="whitegrid", palette="muted")


def _save(fig: plt.Figure, filename: str) -> Path:
    path = CFG.figure_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", path)
    return path


class EDAVisualizer:
    """Orchestrates generation of all required EDA charts."""

    def plot_target_distribution(self, df: pd.DataFrame) -> Path:
        counts = df["Late_Delivery"].value_counts().sort_index()
        labels = ["On Time (0)", "Late (1)"]
        colors = ["#2563EB", "#DC2626"]
        pct = counts / counts.sum() * 100

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
        for bar, pct_val, count in zip(bars, pct.values, counts.values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                         f"{count:,}\n({pct_val:.1f}%)", ha="center", fontsize=10)
        axes[0].set_title("Class Distribution: On Time vs Late Deliveries")
        axes[0].set_ylabel("Number of Orders")
        axes[0].set_ylim(0, counts.max() * 1.2)

        axes[1].pie(counts.values, labels=[f"{l}\n{p:.1f}%" for l, p in zip(labels, pct)],
                    colors=colors, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
        axes[1].set_title("Proportion of Late vs On-Time Orders")
        plt.tight_layout()
        return _save(fig, "01_target_distribution.png")

    def plot_numerical_distributions(self, df: pd.DataFrame) -> Path:
        num_cols = CFG.numerical_features
        n = len(num_cols)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            data = df[col].dropna()
            axes[i].hist(data, bins=50, color="#2563EB", alpha=0.7, edgecolor="white")
            axes[i].set_title(f"{col}\n(median={data.median():.1f})")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Univariate Distributions — Numerical Features", fontsize=14, y=1.01)
        plt.tight_layout()
        return _save(fig, "02_numerical_distributions.png")

    def plot_categorical_distributions(self, df: pd.DataFrame) -> Path:
        cat_cols = CFG.categorical_features
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            top_vals = df[col].value_counts().head(15)
            axes[i].barh(top_vals.index[::-1], top_vals.values[::-1], color="#2563EB", alpha=0.8)
            axes[i].set_title(f"Top Categories — {col}")
            axes[i].set_xlabel("Order Count")

        fig.suptitle("Univariate Distributions — Categorical Features (Top 15)", fontsize=14)
        plt.tight_layout()
        return _save(fig, "03_categorical_distributions.png")

    def plot_late_rate_by_state(self, df: pd.DataFrame) -> Path:
        state_rate = (
            df.groupby("customer_state")["Late_Delivery"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "late_rate", "count": "order_count"})
            .sort_values("late_rate", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(state_rate["customer_state"], state_rate["late_rate"] * 100,
               color=plt.cm.RdYlGn_r(state_rate["late_rate"].values), edgecolor="white")
        ax.axhline(y=df["Late_Delivery"].mean() * 100, color="navy", linestyle="--", linewidth=1.5,
                   label=f"Overall avg ({df['Late_Delivery'].mean():.1%})")
        ax.set_title("Late Delivery Rate by Customer Destination State in Brazil")
        ax.set_xlabel("Customer State")
        ax.set_ylabel("Late Delivery Rate (%)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        return _save(fig, "04_late_rate_by_customer_state.png")

    def plot_late_rate_by_category(self, df: pd.DataFrame) -> Path:
        cat_rate = (
            df.groupby("product_category_name_english")["Late_Delivery"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"product_category_name_english": "category", "mean": "late_rate"})
        )
        cat_rate = cat_rate[cat_rate["count"] >= 50].sort_values("late_rate", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(cat_rate["category"][::-1], cat_rate["late_rate"][::-1] * 100,
                color=plt.cm.RdYlGn_r(cat_rate["late_rate"].values[::-1]), edgecolor="white")
        ax.axhline(y=0, color="navy")
        ax.set_title("Top 20 Product Categories by Late Delivery Rate (Min. 50 orders)")
        ax.set_xlabel("Late Delivery Rate (%)")
        plt.tight_layout()
        return _save(fig, "05_late_rate_by_category.png")

    def plot_delivery_days_distribution(self, df: pd.DataFrame) -> Path:
        df_plot = df[["estimated_delivery_days", "Late_Delivery"]].dropna().copy()
        df_plot["Status"] = df_plot["Late_Delivery"].map({0: "On Time", 1: "Late"})

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for status, color in [("On Time", "#2563EB"), ("Late", "#DC2626")]:
            data = df_plot[df_plot["Status"] == status]["estimated_delivery_days"]
            axes[0].hist(data, bins=40, alpha=0.5, color=color, label=status, density=True)
        axes[0].set_title("Estimated Delivery Days by Status")
        axes[0].set_xlabel("Estimated Days")
        axes[0].legend()

        df_plot.boxplot(column="estimated_delivery_days", by="Status", ax=axes[1],
                        boxprops=dict(color="navy"), medianprops=dict(color="red", linewidth=2))
        axes[1].set_title("Estimated Delivery Days — Boxplot")
        plt.suptitle("")
        plt.tight_layout()
        return _save(fig, "06_estimated_days_distribution.png")

    def plot_freight_vs_late(self, df: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for status, color in [("On Time", "#2563EB"), ("Late", "#DC2626")]:
            mask = df["Late_Delivery"] == (1 if status == "Late" else 0)
            axes[0].hist(df.loc[mask, "freight_value"].clip(0, 200), bins=40,
                         alpha=0.5, color=color, label=status, density=True)
            axes[1].hist(df.loc[mask, "geo_distance_km"].clip(0, 4000), bins=40,
                         alpha=0.5, color=color, label=status, density=True)

        axes[0].set_title("Freight Value Distribution by Status")
        axes[0].set_xlabel("Freight Value (R$)")
        axes[0].legend()

        axes[1].set_title("Geographic Distance (km) Distribution")
        axes[1].set_xlabel("Seller-to-Customer Distance (km)")
        axes[1].legend()

        plt.tight_layout()
        return _save(fig, "07_freight_geo_vs_late.png")

    def plot_monthly_late_rate(self, df: pd.DataFrame) -> Path:
        monthly = (
            df.groupby("purchase_month")["Late_Delivery"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"purchase_month": "month", "mean": "late_rate", "count": "orders"})
        )
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda x: month_names[x - 1])

        fig, ax1 = plt.subplots(figsize=(11, 4))
        ax1.bar(monthly["month_name"], monthly["late_rate"] * 100, color="#DC2626", alpha=0.7)
        ax1.set_xlabel("Purchase Month")
        ax1.set_ylabel("Late Delivery Rate (%)", color="#DC2626")

        ax2 = ax1.twinx()
        ax2.plot(monthly["month_name"], monthly["orders"], "o--", color="#2563EB", linewidth=2)
        ax2.set_ylabel("Order Volume", color="#2563EB")
        ax2.grid(False)

        ax1.set_title("Late Delivery Rate vs Order Volume by Month")
        plt.tight_layout()
        return _save(fig, "08_monthly_late_rate.png")

    def plot_purchase_hour_late_rate(self, df: pd.DataFrame) -> Path:
        hourly = (
            df.groupby("purchase_hour")["Late_Delivery"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"purchase_hour": "hour", "mean": "late_rate", "count": "orders"})
        )
        fig, ax1 = plt.subplots(figsize=(11, 4))
        ax1.bar(hourly["hour"], hourly["late_rate"] * 100, color="#DC2626", alpha=0.7)
        ax1.set_xlabel("Hour of Purchase (0-23)")
        ax1.set_ylabel("Late Rate (%)", color="#DC2626")

        ax2 = ax1.twinx()
        ax2.plot(hourly["hour"], hourly["orders"], "o--", color="#2563EB", linewidth=2)
        ax2.set_ylabel("Order Volume", color="#2563EB")
        ax2.grid(False)

        ax1.set_title("Late Delivery Rate by Hour of Purchase")
        plt.xticks(range(0, 24))
        plt.tight_layout()
        return _save(fig, "09_purchase_hour_late_rate.png")

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> Path:
        num_cols = CFG.numerical_features + ["Late_Delivery"]
        corr_df = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(13, 10))
        mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
        sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 8})
        ax.set_title("Pearson Correlation Heatmap — Numerical Features")
        plt.tight_layout()
        return _save(fig, "10_correlation_heatmap.png")

    def plot_seller_state_late_rate(self, df: pd.DataFrame) -> Path:
        seller_rate = (
            df.groupby("seller_state")["Late_Delivery"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "late_rate", "count": "count"})
        )
        seller_rate = seller_rate[seller_rate["count"] >= 30].sort_values("late_rate", ascending=False)

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar(seller_rate["seller_state"], seller_rate["late_rate"] * 100,
               color=plt.cm.RdYlGn_r(seller_rate["late_rate"].values), edgecolor="white")
        ax.set_title("Late Delivery Rate by Seller Origin State (Min. 30 orders)")
        ax.set_xlabel("Seller State")
        ax.set_ylabel("Late Delivery Rate (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return _save(fig, "11_late_rate_by_seller_state.png")

    def plot_payment_type_late_rate(self, df: pd.DataFrame) -> Path:
        pay_rate = (
            df.groupby("payment_type")["Late_Delivery"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "late_rate", "count": "count"})
            .sort_values("late_rate", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#DC2626", "#F59E0B", "#10B981", "#6366F1"]
        ax.bar(pay_rate["payment_type"], pay_rate["late_rate"] * 100,
               color=colors[:len(pay_rate)], edgecolor="white")
        ax.set_title("Late Delivery Rate by Payment Instrument")
        ax.set_xlabel("Payment Type")
        ax.set_ylabel("Late Rate (%)")
        plt.tight_layout()
        return _save(fig, "12_late_rate_by_payment_type.png")

    def generate_all(self, merged_df: pd.DataFrame, featured_df: pd.DataFrame) -> Dict[str, Path]:
        logger.info("Generating complete EDA visual suites...")
        return {
            "target_distribution":         self.plot_target_distribution(featured_df),
            "numerical_distributions":     self.plot_numerical_distributions(featured_df),
            "categorical_distributions":   self.plot_categorical_distributions(featured_df),
            "late_rate_by_customer_state": self.plot_late_rate_by_state(featured_df),
            "late_rate_by_category":       self.plot_late_rate_by_category(featured_df),
            "delivery_days_distribution":  self.plot_delivery_days_distribution(featured_df),
            "freight_vs_late":             self.plot_freight_vs_late(featured_df),
            "monthly_late_rate":           self.plot_monthly_late_rate(featured_df),
            "purchase_hour_late_rate":     self.plot_purchase_hour_late_rate(featured_df),
            "correlation_heatmap":         self.plot_correlation_heatmap(featured_df),
            "seller_state_late_rate":      self.plot_seller_state_late_rate(featured_df),
            "payment_type_late_rate":      self.plot_payment_type_late_rate(featured_df),
        }

