"""
app/dashboard.py
----------------
Main Streamlit application for the E-commerce Late Delivery Prediction System.
Developed as part of the Machine Learning training program at NextGen Institute.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.prediction import Predictor
from app.components import CUSTOM_CSS, render_header, render_kpi_row, render_prediction_result

# Page configuration
st.set_page_config(
    page_title="Late Delivery Predictor | NextGen Institute",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the pre-trained Pickle pipeline once."""
    try:
        return Predictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_comparison_data():
    """Load model comparison CSV if available."""
    path = CFG.reports_path("model_comparison.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_sample_dataset():
    """Load a sample of processed data for stats & exploration."""
    path = CFG.processed_data_path("featured_dataset")
    if path.exists():
        return pd.read_csv(path, nrows=5000)
    return None


def main():
    # Sidebar Navigation
    st.sidebar.title("📦 Logistics Intelligence")
    st.sidebar.caption("E-Commerce Delivery Delay Warning System")

    nav_options = [
        "🏠 Home Page",
        "📊 Dataset Overview",
        "🔍 Exploratory Data Analysis",
        "📈 Model Performance & Evaluation",
        "🔮 Single Order Prediction",
        "📁 Batch Order Prediction",
    ]
    page = st.sidebar.radio("Navigate System", nav_options)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Institution:** NextGen Institute  
        **Course:** Machine Learning Track  
        **Author:** Senior ML Engineering Team  
        **Dataset:** Olist Brazilian E-Commerce  
        """
    )

    # -------------------------------------------------------------------------
    # 1. HOME PAGE
    # -------------------------------------------------------------------------
    if page == "🏠 Home Page":
        render_header(
            title="E-Commerce Late Delivery Prediction System",
            subtitle="Predictive Logistics Platform for Brazilian E-Commerce (Olist)",
            institution=CFG.institution,
        )

        st.markdown(
            """
            ### 📌 Project Background & Executive Summary
            In modern e-commerce, on-time delivery is the cornerstone of customer retention, brand trust, and operational efficiency. When orders miss their estimated delivery windows, e-commerce platforms suffer:
            - **High customer churn & lifetime value loss**
            - **Direct financial burden:** Refunds, reshipments, and customer support ticket overhead
            - **Severe platform rating degradation:** Late deliveries in the Olist dataset strongly correlate with 1-star reviews

            ### 🎯 Core Objectives
            1. **Early Risk Detection:** Classify incoming orders at the exact moment of purchase as either **On-Time (0)** or **Late (1)** before the carrier receives the parcel.
            2. **Data Leakage Immunity:** Ensure zero future information (e.g. carrier transit timestamps, post-dispatch scans, reviews) leaks into model features.
            3. **Operational Decision Support:** Provide supply chain dispatchers with actionable risk levels (Low, Medium, High) and clear intervention recommendations.
            4. **Production Architecture:** Modular Object-Oriented code, automated Scikit-Learn preprocessing pipeline, model serialization via Pickle, and zero-downtime inference.
            """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📦 **Dataset Scope**\n\n~100,000 real-world customer orders across 27 Brazilian states from 2016 to 2018.")
        with col2:
            st.warning("⏱️ **Prediction Point**\n\nInstantaneous inference at checkout (`order_purchase_timestamp`).")
        with col3:
            st.success("🤖 **Ensemble Machine Learning**\n\nTrained, tuned, and validated across 6 classification algorithms with strict stratified isolation.")

    # -------------------------------------------------------------------------
    # 2. DATASET OVERVIEW
    # -------------------------------------------------------------------------
    elif page == "📊 Dataset Overview":
        render_header(title="Dataset Inspection & Key Statistics", subtitle="Brazilian E-Commerce Public Dataset by Olist")
        df_sample = load_sample_dataset()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Delivered Orders", "96,470")
        col2.metric("Late Delivery Rate", "8.11%")
        col3.metric("Engineered Features", "18 Features")
        col4.metric("Relational Tables", "9 Tables")

        st.markdown("### 📋 Sample Model-Ready Features")
        if df_sample is not None:
            st.dataframe(df_sample.head(20), use_container_width=True)
        else:
            st.info("Run `python train.py` to generate the processed dataset.")

        st.markdown("### 🗺️ Target Variable Breakdown")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            labels = ["On Time (Class 0)", "Late (Class 1)"]
            values = [88644, 7826]
            fig = px.pie(
                values=values,
                names=labels,
                color_discrete_sequence=["#2563eb", "#ef4444"],
                hole=0.4,
                title="Class Distribution (Late_Delivery)",
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown(
                """
                #### Key Class Imbalance Takeaways:
                - **Class 0 (On-Time):** 91.89% (88,644 orders)
                - **Class 1 (Late):** 8.11% (7,826 orders)
                - **Critical ML Decision:** Accuracy is deceptive in this domain (a naive model predicting all on-time scores ~92% accuracy while detecting 0 delays).
                - **Strategic Metrics:** Optimization focuses heavily on **F1-Score**, **Recall**, and **ROC-AUC** with class weighting.
                """
            )

    # -------------------------------------------------------------------------
    # 3. EXPLORATORY DATA ANALYSIS
    # -------------------------------------------------------------------------
    elif page == "🔍 Exploratory Data Analysis":
        render_header(title="Exploratory Data Analysis (EDA)", subtitle="Uncovering Spatial, Temporal, and Structural Drivers of Delays")

        tabs = st.tabs(["Geographic Delays", "Product & Category", "Temporal Factors", "Pricing & Freight"])

        figures_dir = CFG.figures_dir

        with tabs[0]:
            st.subheader("Geographic Drivers of Late Delivery")
            f1 = figures_dir / "04_late_rate_by_customer_state.png"
            if f1.exists():
                st.image(str(f1), caption="Late Delivery Rate by Brazilian Customer State", use_container_width=True)
            f2 = figures_dir / "11_late_rate_by_seller_state.png"
            if f2.exists():
                st.image(str(f2), caption="Late Delivery Rate by Seller Origin State", use_container_width=True)

        with tabs[1]:
            st.subheader("Product Category Discrepancies")
            f3 = figures_dir / "05_late_rate_by_category.png"
            if f3.exists():
                st.image(str(f3), caption="Top 20 Product Categories with Highest Late Delivery Rates", use_container_width=True)

        with tabs[2]:
            st.subheader("Seasonality & Time-of-Purchase Patterns")
            f4 = figures_dir / "08_monthly_late_rate.png"
            if f4.exists():
                st.image(str(f4), caption="Monthly Order Volume vs Delay Rate", use_container_width=True)
            f5 = figures_dir / "09_purchase_hour_late_rate.png"
            if f5.exists():
                st.image(str(f5), caption="Late Delivery Rates Across 24-Hour Purchase Window", use_container_width=True)

        with tabs[3]:
            st.subheader("Freight Value and Correlation Structure")
            f6 = figures_dir / "07_freight_geo_vs_late.png"
            if f6.exists():
                st.image(str(f6), caption="Freight Cost & Distance vs Late Delivery", use_container_width=True)
            f7 = figures_dir / "10_correlation_heatmap.png"
            if f7.exists():
                st.image(str(f7), caption="Correlation Matrix for Numerical Features", use_container_width=True)

    # -------------------------------------------------------------------------
    # 4. MODEL PERFORMANCE & EVALUATION
    # -------------------------------------------------------------------------
    elif page == "📈 Model Performance & Evaluation":
        render_header(title="Model Comparison & Evaluation", subtitle="Rigorous Empirical Validation on Holdout Datasets")

        comp_df = load_comparison_data()
        if comp_df is not None:
            st.subheader("📊 Comparative Algorithm Benchmark (Dev Set)")
            st.dataframe(
                comp_df.style.highlight_max(subset=["accuracy", "precision", "recall", "f1_score", "roc_auc"], color="#dcfce7"),
                use_container_width=True,
            )

            best_row = comp_df.iloc[0]
            st.markdown(f"### 🏆 Champion Model Selected: **{best_row['model_name']}**")
            render_kpi_row(best_row.to_dict())
        else:
            st.info("Evaluation metrics will appear after running `python train.py`.")

        st.markdown("---")
        st.subheader("🔬 Diagnostic Visualizations")
        col_c1, col_c2 = st.columns(2)

        figures_dir = CFG.figures_dir
        with col_c1:
            roc_files = list(figures_dir.glob("roc_curve_*_test.png")) or list(figures_dir.glob("roc_curve_*_dev.png"))
            if roc_files:
                st.image(str(roc_files[0]), caption="ROC Curve Analysis", use_container_width=True)

            pr_files = list(figures_dir.glob("pr_curve_*_test.png")) or list(figures_dir.glob("pr_curve_*_dev.png"))
            if pr_files:
                st.image(str(pr_files[0]), caption="Precision-Recall Curve Analysis", use_container_width=True)

        with col_c2:
            cm_files = list(figures_dir.glob("confusion_matrix_*_test.png")) or list(figures_dir.glob("confusion_matrix_*_dev.png"))
            if cm_files:
                st.image(str(cm_files[0]), caption="Normalised Confusion Matrix", use_container_width=True)

            fi_files = list(figures_dir.glob("feature_importance_*.png"))
            if fi_files:
                st.image(str(fi_files[0]), caption="Feature Importance Ranking", use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. SINGLE ORDER PREDICTION
    # -------------------------------------------------------------------------
    elif page == "🔮 Single Order Prediction":
        render_header(title="Interactive Order Delay Predictor", subtitle="Real-Time Inference via Serialized Pipeline")

        predictor = load_predictor()
        if predictor is None:
            st.error("Model file `models/best_model.pkl` not found. Please run `python train.py` first.")
            return

        st.markdown("#### Enter Order Specification")
        with st.form("single_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 📅 Temporal Parameters")
                purchase_month = st.selectbox("Purchase Month", list(range(1, 13)), index=10)
                purchase_day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)), index=4)
                purchase_hour = st.slider("Purchase Hour (0-23)", 0, 23, 14)
                estimated_delivery_days = st.number_input("Estimated Delivery Promise (Days)", 1, 60, 16)

            with col2:
                st.markdown("##### 📦 Product & Basket Details")
                price = st.number_input("Item Price (R$)", min_value=1.0, max_value=10000.0, value=120.0)
                freight_value = st.number_input("Freight Charge (R$)", min_value=0.0, max_value=1000.0, value=25.0)
                product_weight_g = st.number_input("Product Weight (grams)", min_value=10.0, max_value=50000.0, value=750.0)
                product_volume_cm3 = st.number_input("Product Volume (cm³)", min_value=10.0, max_value=100000.0, value=6000.0)
                n_items = st.number_input("Number of Items in Order", min_value=1, max_value=20, value=1)
                n_sellers = st.number_input("Distinct Sellers in Order", min_value=1, max_value=10, value=1)

            with col3:
                st.markdown("##### 📍 Geography & Payment")
                customer_state = st.selectbox("Customer Destination State", [
                    "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO",
                    "PE", "CE", "PA", "MT", "MA", "MS", "PB", "PI", "RN", "AL",
                    "SE", "TO", "RO", "AM", "AC", "AP", "RR"
                ], index=1)
                seller_state = st.selectbox("Seller Origin State", [
                    "SP", "PR", "MG", "RJ", "SC", "RS", "GO", "DF", "ES", "BA",
                    "PE", "CE", "RN", "MS", "PB", "MT", "SE", "PA", "AM", "RO"
                ], index=0)
                geo_distance_km = st.number_input("Geographic Distance (km)", min_value=1.0, max_value=5000.0, value=450.0)
                payment_type = st.selectbox("Payment Type", ["credit_card", "boleto", "voucher", "debit_card"], index=0)
                payment_installments = st.number_input("Payment Installments", 1, 24, 3)
                category_list = [
                    "bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor",
                    "computers_accessories", "housewares", "watches_gifts", "telephony",
                    "auto", "toys", "cool_stuff", "electronics", "baby", "perfumery", "unknown"
                ]
                product_category_name_english = st.selectbox("Product Category", category_list, index=0)

            submitted = st.form_submit_button("🚀 Run Real-Time Prediction")

        if submitted:
            total_payment_value = (price + freight_value) * n_items
            freight_to_price_ratio = freight_value / max(price, 1e-5)

            order_payload = {
                "purchase_month": purchase_month,
                "purchase_day_of_week": purchase_day_of_week,
                "purchase_hour": purchase_hour,
                "estimated_delivery_days": estimated_delivery_days,
                "product_weight_g": product_weight_g,
                "product_volume_cm3": product_volume_cm3,
                "freight_value": freight_value,
                "price": price,
                "freight_to_price_ratio": freight_to_price_ratio,
                "n_items": n_items,
                "n_sellers": n_sellers,
                "payment_installments": payment_installments,
                "total_payment_value": total_payment_value,
                "geo_distance_km": geo_distance_km,
                "product_category_name_english": product_category_name_english,
                "seller_state": seller_state,
                "customer_state": customer_state,
                "payment_type": payment_type,
            }

            with st.spinner("Analyzing shipping route & order parameters..."):
                res = predictor.predict_single(order_payload)
                render_prediction_result(res)

    # -------------------------------------------------------------------------
    # 6. BATCH ORDER PREDICTION
    # -------------------------------------------------------------------------
    elif page == "📁 Batch Order Prediction":
        render_header(title="Batch Order Processing", subtitle="High-Throughput Scoring for Warehouse Logistics")

        predictor = load_predictor()
        if predictor is None:
            st.error("Model file not found. Run training pipeline first.")
            return

        st.markdown(
            """
            Upload a CSV containing order records. The pipeline will automatically:
            1. Validate column signatures
            2. Apply imputations and scalers
            3. Append `predicted_class`, `late_probability`, and `risk_level`
            """
        )

        uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])

        # Provide a sample template download
        df_sample = load_sample_dataset()
        if df_sample is not None:
            template_df = df_sample.drop(columns=[CFG.target_column], errors="ignore").head(10)
            csv_bytes = template_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Sample Template CSV", csv_bytes, "batch_template.csv", "text/csv")

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(input_df)} records for inference.")

                with st.spinner("Executing batch predictions..."):
                    results_df = predictor.predict_batch(input_df)

                st.success("Batch Prediction Complete!")
                st.dataframe(results_df.head(50), use_container_width=True)

                out_csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Download Annotated Predictions CSV",
                    out_csv,
                    "predicted_orders.csv",
                    "text/csv",
                )
            except Exception as ex:
                st.error(f"Error processing batch: {ex}")


if __name__ == "__main__":
    main()

