"""
UPI-SHIELD dashboard reusable UI components.

Every function here renders a piece of UI from data it is given — none
of them call the API or invent values. They format and display
what app/dashboard.py passes in (which ultimately comes from the
FastAPI /predict response or from local result files).
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Risk band styling — mirrors models/risk_scoring_config.json risk_bands.
# ---------------------------------------------------------------------------

RISK_BAND_COLORS = {
    "LOW": "#10b981",       # Emerald
    "MODERATE": "#f59e0b",  # Amber
    "HIGH": "#f97316",      # Orange
    "CRITICAL": "#ef4444",  # Crimson
}

DECISION_COLORS = {
    "FRAUD": "#ef4444",
    "LEGITIMATE": "#10b981",
}


def risk_band_color(band: str) -> str:
    """Return hex color for a risk band."""
    return RISK_BAND_COLORS.get(str(band).upper(), "#94a3b8")


def decision_color(decision: str) -> str:
    """Return hex color for a decision."""
    return DECISION_COLORS.get(str(decision).upper(), "#94a3b8")


# ---------------------------------------------------------------------------
# Global theming & custom CSS
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Inject the dashboard's shared executive stylesheet. Call once per page load."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Base content typography - preserve Streamlit icon ligatures */
        html, body, p, label, input, select, textarea, .stMarkdown, .stSelectbox, .stNumberInput, .stTextInput, .stRadio, .stButton button {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            letter-spacing: -0.01em;
        }

        /* Protect Streamlit UI Material Symbols (sidebar collapse arrow, chevron, headers, etc.) */
        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons,
        button[kind="header"] span,
        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="collapsedControl"] span,
        [data-testid="stHeader"] span,
        header span[class*="material"],
        header button span {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            font-style: normal !important;
        }

        code, pre, .mono-font {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Container constraints and breathing room */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            max-width: 1280px;
        }

        /* Header typography with subtle gradient accents */
        h1 {
            font-size: 2.1rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            margin-bottom: 0.2rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.6rem !important;
        }

        h3 {
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
        }

        /* Hero / Page header banner */
        .shield-hero-header {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(12px);
        }

        .shield-hero-title {
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .shield-hero-subtitle {
            font-size: 0.95rem;
            color: #94a3b8;
            margin-top: 0.35rem;
            margin-bottom: 0;
            line-height: 1.4;
        }

        /* Modern card container */
        .shield-card {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
            transition: all 0.2s ease-in-out;
        }

        .shield-card:hover {
            border-color: rgba(255, 255, 255, 0.12);
            box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.45);
        }

        /* Metric cards */
        .shield-kpi-card {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            text-align: left;
            position: relative;
            overflow: hidden;
        }

        .shield-kpi-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #94a3b8;
            margin-bottom: 0.3rem;
        }

        .shield-kpi-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.7rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.1;
        }

        .shield-kpi-sub {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: 0.25rem;
        }

        /* Badges */
        .shield-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 0.03em;
            border: 1px solid rgba(127,127,127,0.25);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }

        .shield-badge-dot {
            height: 7px;
            width: 7px;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px currentColor;
        }

        .shield-section-label {
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.74rem;
            font-weight: 700;
            color: #94a3b8;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        /* Decision Banner */
        .shield-decision-banner {
            border-radius: 14px;
            padding: 1.1rem 1.5rem;
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-align: center;
            border: 1.5px solid rgba(127,127,127,0.3);
            margin-top: 1rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 8px 24px -4px rgba(0,0,0,0.35);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }

        .shield-decision-subtext {
            font-size: 0.85rem;
            font-weight: 500;
            opacity: 0.85;
            margin-top: 0.2rem;
        }

        /* Pipeline visualizer */
        .shield-pipeline-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.6rem;
            align-items: center;
            margin: 1.2rem 0;
        }

        .shield-pipeline-step {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.09);
            padding: 0.8rem 0.6rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.82rem;
            text-align: center;
            color: #e2e8f0;
            position: relative;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.2s;
        }

        .shield-pipeline-step:hover {
            border-color: #6366f1;
            transform: translateY(-2px);
        }

        .shield-pipeline-num {
            font-size: 0.68rem;
            color: #818cf8;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .shield-caption {
            font-size: 0.82rem;
            color: #94a3b8;
            line-height: 1.4;
        }

        /* Quick presets pill buttons */
        .preset-badge {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #cbd5e1;
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            display: inline-block;
        }

        /* Sidebar refinements */
        [data-testid="stSidebar"] {
            background-color: #0b1120;
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }

        /* Custom scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.6);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(100, 116, 139, 0.4);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(100, 116, 139, 0.7);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# API status badge
# ---------------------------------------------------------------------------

def render_api_status_badge(status: bool | None) -> None:
    """Render a modern CONNECTED / DISCONNECTED / CHECKING status badge."""

    if status is True:
        color = "#10b981"
        label = "API CONNECTED"
        pulse = "0 0 10px rgba(16, 185, 129, 0.5)"
    elif status is False:
        color = "#ef4444"
        label = "API DISCONNECTED"
        pulse = "0 0 10px rgba(239, 68, 68, 0.5)"
    else:
        color = "#94a3b8"
        label = "CHECKING..."
        pulse = "none"

    st.markdown(
        f"""
        <div class="shield-badge" style="color:{color}; border-color:{color}55; background: {color}12; width: 100%; justify-content: center;">
            <span class="shield-badge-dot" style="background:{color}; box-shadow:{pulse};"></span>
            <span>{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Risk Analyzer result components
# ---------------------------------------------------------------------------

def render_risk_gauge(risk_score: int, risk_band: str) -> go.Figure:
    """Build a precision 0-100 radial risk gauge figure."""

    color = risk_band_color(risk_band)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={
                "suffix": " / 100",
                "font": {"size": 42, "family": "JetBrains Mono, monospace", "color": "#f8fafc"},
            },
            title={
                "text": f"<b>RISK BAND: <span style='color:{color}'>{risk_band}</span></b>",
                "font": {"size": 15, "family": "Plus Jakarta Sans, sans-serif", "color": "#94a3b8"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1.5,
                    "tickcolor": "#475569",
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0", "25", "50", "75", "100"],
                    "tickfont": {"size": 11, "color": "#94a3b8", "family": "JetBrains Mono"},
                },
                "bar": {"color": color, "thickness": 0.38},
                "bgcolor": "rgba(30, 41, 59, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(16, 185, 129, 0.12)"},
                    {"range": [25, 50], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [50, 75], "color": "rgba(249, 115, 22, 0.12)"},
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.16)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.88,
                    "value": risk_score,
                },
            },
        )
    )

    fig.update_layout(
        height=270,
        margin=dict(l=25, r=25, t=50, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans, sans-serif"},
    )

    return fig


def render_result_metrics(result: dict[str, Any]) -> None:
    """Render the executive metric cards and decision banner for a /predict response."""

    band = result["risk_band"]
    decision = result["decision"]
    d_color = decision_color(decision)
    b_color = risk_band_color(band)

    prob = result["fraud_probability"]
    score = result["risk_score"]
    threshold = result["decision_threshold"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Fraud Probability</div>
                <div class="shield-kpi-value">{prob:.2%}</div>
                <div class="shield-kpi-sub">Raw model output</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Risk Score</div>
                <div class="shield-kpi-value">{score}<span style="font-size:1rem;color:#64748b;">/100</span></div>
                <div class="shield-kpi-sub">Adaptive scale</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Risk Band</div>
                <div style="margin-top: 0.15rem; margin-bottom: 0.25rem;">
                    <span class="shield-badge" style="color:{b_color}; border-color:{b_color}66; background:{b_color}18; font-size: 0.95rem;">
                        <span class="shield-badge-dot" style="background:{b_color};"></span>
                        {band}
                    </span>
                </div>
                <div class="shield-kpi-sub">Severity Tier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Cost-Aware Threshold</div>
                <div class="shield-kpi-value">{threshold:.2f}</div>
                <div class="shield-kpi-sub">Optimum τ (FP:1, FN:10)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # High-impact decision banner
    decision_icon = "🚨" if decision == "FRAUD" else "🛡️"
    decision_desc = (
        f"Fraud probability ({prob:.2%}) exceeds cost-optimal threshold ({threshold:.2f}). Step-up verification or block recommended."
        if decision == "FRAUD"
        else f"Fraud probability ({prob:.2%}) is below cost-optimal threshold ({threshold:.2f}). Transaction cleared for standard processing."
    )

    st.markdown(
        f"""
        <div class="shield-decision-banner"
             style="color:{d_color};
                    background:{d_color}14;
                    border-color:{d_color}66;
                    box-shadow: 0 0 20px {d_color}22;">
            <div>
                <span style="font-size:1.5rem; vertical-align:middle; margin-right:0.4rem;">{decision_icon}</span>
                DECISION: {decision}
                <div class="shield-decision-subtext">{decision_desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Probability source: `{result['probability_source']}` • "
        "Calculated by deployed XGBoost model • "
        "Cost-aware decision policy τ = 0.09"
    )


def render_shap_compact(explanations: list[dict[str, Any]], top_n: int = 5) -> None:
    """Compact risk factor cards for the Risk Analyzer page."""

    if not explanations:
        st.info("No explanation factors were returned for this transaction.")
        return

    for item in explanations[:top_n]:
        shap_val = item["shap_value"]
        is_pos = shap_val > 0
        color = "#ef4444" if is_pos else "#10b981"
        icon = "🔺 Increases Risk" if is_pos else "🔻 Decreases Risk"

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.6rem 0.9rem; margin-bottom:0.4rem;
                        background:rgba(15, 23, 42, 0.6);
                        border:1px solid rgba(255,255,255,0.06);
                        border-left: 3px solid {color};
                        border-radius:8px;">
                <div style="display:flex; flex-direction:column;">
                    <span style="font-weight:600; color:#e2e8f0; font-size:0.92rem;">
                        {item['feature']}
                    </span>
                    <span style="font-size:0.75rem; color:#94a3b8; margin-top:0.1rem;">
                        {item.get('message', '')}
                    </span>
                </div>
                <div style="text-align:right;">
                    <span class="mono-font" style="color:{color}; font-weight:700; font-size:0.95rem;">
                        {shap_val:+.4f}
                    </span>
                    <div style="font-size:0.7rem; color:{color}; font-weight:600;">
                        {icon}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_shap_full(explanations: list[dict[str, Any]]) -> None:
    """Full SHAP bar chart + narrative factors table for the Explainability page."""

    if not explanations:
        st.info(
            "💡 No transaction has been analyzed in this session yet. "
            "Navigate to **Risk Analyzer**, submit a transaction, and return here "
            "to inspect its complete SHAP feature attribution breakdown."
        )
        return

    # Summary metric deck for explanations
    pos_factors = [x for x in explanations if x["shap_value"] > 0]
    neg_factors = [x for x in explanations if x["shap_value"] <= 0]
    sum_pos = sum(x["shap_value"] for x in pos_factors)
    sum_neg = sum(x["shap_value"] for x in neg_factors)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Factors Evaluated</div>
                <div class="shield-kpi-value">{len(explanations)}</div>
                <div class="shield-kpi-sub">SHAP TreeExplainer Attribution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Risk Escalation Drivers</div>
                <div class="shield-kpi-value" style="color:#ef4444;">+{sum_pos:.3f}</div>
                <div class="shield-kpi-sub">{len(pos_factors)} factors pushing towards FRAUD</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Risk Mitigating Drivers</div>
                <div class="shield-kpi-value" style="color:#10b981;">{sum_neg:.3f}</div>
                <div class="shield-kpi-sub">{len(neg_factors)} factors confirming legitimacy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="shield-section-label">Feature Attribution Waterfall (SHAP Values)</div>', unsafe_allow_html=True)

    ordered = sorted(explanations, key=lambda item: item["shap_value"])

    features = [item["feature"] for item in ordered]
    values = [item["shap_value"] for item in ordered]
    colors = ["#ef4444" if val > 0 else "#10b981" for val in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
            ),
            text=[f" {v:+.4f}" for v in values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=12, color="#cbd5e1"),
            hovertemplate="<b>%{y}</b><br>SHAP Attribution: <b>%{x:+.4f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        height=max(340, 42 * len(features)),
        margin=dict(l=20, r=45, t=25, b=20),
        xaxis=dict(
            title=dict(
                text="SHAP Attribution (Log-odds Impact)",
                font=dict(family="Plus Jakarta Sans", size=12, color="#94a3b8"),
            ),
            tickfont=dict(family="JetBrains Mono", size=11, color="#94a3b8"),
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.25)",
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            tickfont=dict(family="Plus Jakarta Sans", size=12, color="#e2e8f0"),
            gridcolor="rgba(255, 255, 255, 0.04)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="shield-caption">'
        "🔴 <b>Positive SHAP values</b> shift model prediction towards Fraud risk. "
        "🟢 <b>Negative SHAP values</b> anchor prediction towards Legitimate behavior."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<div class="shield-section-label">Detailed Factor Explanations & Evidence</div>', unsafe_allow_html=True)

    table_rows = [
        {
            "Risk Factor": item["feature"],
            "SHAP Impact": f"{item['shap_value']:+.5f}",
            "Direction": item["direction"].replace("_", " ").title(),
            "Diagnostic Insight": item["message"],
        }
        for item in explanations
    ]

    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    with st.expander("🔍 Technical Feature Mapping & Internal Tokens"):
        tech_rows = [
            {
                "Readable Feature": item["feature"],
                "Internal Model Token": item["technical_feature"],
                "SHAP Impact": round(item["shap_value"], 6),
            }
            for item in explanations
        ]
        st.dataframe(tech_rows, use_container_width=True, hide_index=True)

    st.info(
        "ℹ️ **Interpretability Notice:** SHAP values quantify additive feature contributions "
        "to the XGBoost decision boundary for this specific transaction. They describe model "
        "attribution and should not be interpreted as definitive causal proof."
    )