"""
app/components.py
-----------------
Reusable UI components and custom CSS styling for the Streamlit dashboard.
"""

from __future__ import annotations

from typing import Any, Dict
import streamlit as st

CUSTOM_CSS = """
<style>
    .main { background-color: #f8fafc; }
    .brand-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .brand-header h1 { color: #ffffff !important; font-weight: 800; margin-bottom: 0.5rem; }
    .brand-header p { color: #e0e7ff; font-size: 1.1rem; margin-bottom: 0; }
    .badge-institute {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.2);
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .kpi-card {
        background: white;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .kpi-title { font-size: 0.875rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; color: #0f172a; }
    .kpi-subtitle { font-size: 0.8rem; color: #94a3b8; }
    .pred-card-on-time {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #065f46;
        margin-top: 1.5rem;
    }
    .pred-card-late {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: #991b1b;
        margin-top: 1.5rem;
    }
</style>
"""


def render_header(
    title: str = "E-Commerce Late Delivery Prediction System",
    subtitle: str = "Real-Time Machine Learning Logistics Intelligence Platform",
    institution: str = "NextGen Institute",
) -> None:
    """Render top branding banner."""
    st.markdown(
        f"""
        <div class="brand-header">
            <span class="badge-institute">🎓 {institution} — Final Project</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(metrics: Dict[str, Any]) -> None:
    """Render a 5-column KPI metric card row."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Accuracy</div>
                <div class="kpi-value">{metrics.get('accuracy', 0.0):.2%}</div>
                <div class="kpi-subtitle">Overall correctness</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #10b981;">
                <div class="kpi-title">Precision</div>
                <div class="kpi-value">{metrics.get('precision', 0.0):.2%}</div>
                <div class="kpi-subtitle">When flagged late, is it late?</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #f59e0b;">
                <div class="kpi-title">Recall</div>
                <div class="kpi-value">{metrics.get('recall', 0.0):.2%}</div>
                <div class="kpi-subtitle">Late orders caught</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #8b5cf6;">
                <div class="kpi-title">F1-Score</div>
                <div class="kpi-value">{metrics.get('f1_score', 0.0):.3f}</div>
                <div class="kpi-subtitle">Harmonic mean (Prec & Rec)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color: #ec4899;">
                <div class="kpi-title">ROC-AUC</div>
                <div class="kpi-value">{metrics.get('roc_auc', 0.0):.3f}</div>
                <div class="kpi-subtitle">Discrimination power</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_result(result: Dict[str, Any]) -> None:
    """Render prediction outcome card with badge and business action."""
    is_late = result["predicted_class"] == 1
    prob = result["late_probability"]
    risk = result["risk_level"]
    msg = result["message"]

    card_class = "pred-card-late" if is_late else "pred-card-on-time"
    icon = "🚨" if is_late else "✅"
    title = "PREDICTION: HIGH RISK OF LATE DELIVERY" if is_late else "PREDICTION: ON TIME DELIVERY"

    st.markdown(
        f"""
        <div class="{card_class}">
            <h3 style="margin-top:0; color: inherit;">{icon} {title}</h3>
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>Risk Level:</strong> {risk} ({prob:.1%} estimated delay probability)</p>
            <p style="margin-bottom: 0;"><strong>Operational Guidance:</strong> {msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.progress(prob, text=f"Estimated Late Delivery Probability: {prob:.1%}")


# When executed directly as entrypoint on Streamlit Cloud, redirect to dashboard main
if __name__ == "__main__":
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from app.dashboard import main as _dashboard_main

    _dashboard_main()
