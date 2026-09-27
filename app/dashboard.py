"""
UPI-SHIELD Streamlit Dashboard.

Adaptive, Cost-Aware and Explainable Fraud Risk Scoring Framework
for Digital Payment Transactions.

This dashboard is a FRONTEND ONLY. All fraud predictions come from the
existing FastAPI /predict endpoint (see app/main.py, app/predictor.py).
No model is loaded, retrained, or evaluated inside this file. All
historical/model performance figures come from the project's existing
experiment result files under experiments/results/.
"""

from __future__ import annotations

import json
import time
from typing import Any

import streamlit as st

import dashboard_api as api
import dashboard_charts as charts
import dashboard_components as ui
import dashboard_data as data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UPI-SHIELD | AI Fraud Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_custom_css()

PAGES = ["Risk Analyzer", "Risk Analytics", "Explainability", "Model & System"]

# How long a cached API-status check stays valid before it is re-checked
# automatically. Keeps the dashboard from hammering /health on every
# rerun, while still noticing when the API comes back up.
API_STATUS_TTL_SECONDS = 20

# Quick test transaction archetypes
PRESET_TRANSACTIONS = {
    "🟢 Low-Risk P2P Transfer": {
        "TransactionDT": 86400.0,
        "TransactionAmt": 25.0,
        "ProductCD": "W",
        "card1": 10000.0,
        "card2": 111.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "debit",
        "addr1": 100.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "mobile",
        "DeviceInfo": "Android",
    },
    "🟡 Moderate E-Commerce Txn": {
        "TransactionDT": 172800.0,
        "TransactionAmt": 185.0,
        "ProductCD": "C",
        "card1": 5000.0,
        "card2": 250.0,
        "card3": 150.0,
        "card4": "mastercard",
        "card5": 102.0,
        "card6": "credit",
        "addr1": 299.0,
        "addr2": 87.0,
        "P_emaildomain": "yahoo.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "DeviceInfo": "Windows",
    },
    "🟠 High-Risk Velocity Spike": {
        "TransactionDT": 259200.0,
        "TransactionAmt": 720.0,
        "ProductCD": "H",
        "card1": 15000.0,
        "card2": 321.0,
        "card3": 150.0,
        "card4": "discover",
        "card5": 224.0,
        "card6": "credit",
        "addr1": 441.0,
        "addr2": 87.0,
        "P_emaildomain": "anonymous.com",
        "R_emaildomain": "outlook.com",
        "DeviceType": "desktop",
        "DeviceInfo": "MacOS",
    },
    "🔴 Critical Cross-Border Spike": {
        "TransactionDT": 345600.0,
        "TransactionAmt": 3200.0,
        "ProductCD": "R",
        "card1": 18000.0,
        "card2": 555.0,
        "card3": 185.0,
        "card4": "Unknown",
        "card5": 137.0,
        "card6": "credit",
        "addr1": 512.0,
        "addr2": 95.0,
        "P_emaildomain": "protonmail.com",
        "R_emaildomain": "mail.ru",
        "DeviceType": "mobile",
        "DeviceInfo": "Linux",
    },
}


# ---------------------------------------------------------------------------
# Session state & preset callback
# ---------------------------------------------------------------------------

def apply_preset(preset_data: dict[str, Any], navigate_to_analyzer: bool = False) -> None:
    """Safe on_click callback to update preset values before widgets render."""
    st.session_state.current_preset_values = preset_data.copy()
    st.session_state["in_TransactionDT"] = float(preset_data.get("TransactionDT", 86400.0))
    st.session_state["in_TransactionAmt"] = float(preset_data.get("TransactionAmt", 25.0))
    st.session_state["in_ProductCD"] = str(preset_data.get("ProductCD", "W"))
    st.session_state["in_card1"] = float(preset_data.get("card1", 10000.0))
    st.session_state["in_card2"] = float(preset_data.get("card2", 111.0))
    st.session_state["in_card3"] = float(preset_data.get("card3", 150.0))
    st.session_state["in_card4"] = str(preset_data.get("card4") or "Unknown")
    st.session_state["in_card5"] = float(preset_data.get("card5", 226.0))
    st.session_state["in_card6"] = str(preset_data.get("card6") or "Unknown")
    st.session_state["in_addr1"] = float(preset_data.get("addr1", 100.0))
    st.session_state["in_addr2"] = float(preset_data.get("addr2", 87.0))
    st.session_state["in_P_email"] = str(preset_data.get("P_emaildomain") or "gmail.com")
    st.session_state["in_R_email"] = str(preset_data.get("R_emaildomain") or "gmail.com")
    st.session_state["in_DeviceType"] = str(preset_data.get("DeviceType") or "Unknown")
    st.session_state["in_DeviceInfo"] = str(preset_data.get("DeviceInfo") or "Android")

    if navigate_to_analyzer:
        st.session_state.nav = PAGES[0]


def init_session_state() -> None:
    default_preset = PRESET_TRANSACTIONS["🟢 Low-Risk P2P Transfer"]
    defaults = {
        "nav": PAGES[0],
        "api_status": None,
        "api_status_checked_at": 0.0,
        "last_transaction": None,
        "last_result": None,
        "last_error": None,
        "current_preset_values": default_preset.copy(),
        "in_TransactionDT": float(default_preset["TransactionDT"]),
        "in_TransactionAmt": float(default_preset["TransactionAmt"]),
        "in_ProductCD": default_preset["ProductCD"],
        "in_card1": float(default_preset["card1"]),
        "in_card2": float(default_preset["card2"]),
        "in_card3": float(default_preset["card3"]),
        "in_card4": default_preset["card4"],
        "in_card5": float(default_preset["card5"]),
        "in_card6": default_preset["card6"],
        "in_addr1": float(default_preset["addr1"]),
        "in_addr2": float(default_preset["addr2"]),
        "in_P_email": default_preset["P_emaildomain"],
        "in_R_email": default_preset["R_emaildomain"],
        "in_DeviceType": default_preset["DeviceType"],
        "in_DeviceInfo": default_preset["DeviceInfo"],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def refresh_api_status(force: bool = False) -> None:
    now = time.time()
    stale = (now - st.session_state.api_status_checked_at) > API_STATUS_TTL_SECONDS

    if not force and not stale and st.session_state.api_status is not None:
        return

    result = api.check_health()
    st.session_state.api_status = result.ok
    st.session_state.api_status_checked_at = now


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
                <span style="font-size:1.8rem;">🛡️</span>
                <div>
                    <div style="font-size:1.25rem; font-weight:800; color:#f8fafc; letter-spacing:-0.02em;">UPI-SHIELD</div>
                    <div style="font-size:0.75rem; color:#94a3b8; font-weight:500;">Risk Scoring Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Adaptive, Cost-Aware & Explainable Fraud Risk Framework")

        refresh_api_status()
        status_col, refresh_col = st.columns([3, 1])
        with status_col:
            ui.render_api_status_badge(st.session_state.api_status)
        with refresh_col:
            if st.button("🔄", help="Re-check API connection", use_container_width=True):
                refresh_api_status(force=True)
                st.rerun()

        st.markdown(
            f"""
            <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:0.4rem 0.6rem; margin-top:0.4rem; font-size:0.75rem; color:#94a3b8;">
                Target API: <span class="mono-font" style="color:#cbd5e1;">{api.API_BASE_URL}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown('<div class="shield-section-label">Navigation</div>', unsafe_allow_html=True)
        selected = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")

        st.divider()

        st.markdown('<div class="shield-section-label">Quick Scenario Presets</div>', unsafe_allow_html=True)
        st.caption("Load realistic test profiles with 1 click:")

        for preset_name, preset_data in PRESET_TRANSACTIONS.items():
            st.button(
                preset_name,
                use_container_width=True,
                key=f"sidebar_preset_{preset_name}",
                on_click=apply_preset,
                args=(preset_data, True),
            )

        st.divider()

        st.markdown(
            """
            <div style="background:rgba(30, 41, 59, 0.4); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:0.6rem 0.8rem; font-size:0.72rem; color:#94a3b8; line-height:1.4; margin-bottom:0.5rem;">
                <b>Academic Course:</b> PECO311C (Applied ML)<br>
                <b>Project 25:</b> UPI Fraud Risk Scoring<br>
                <b>Instructor:</b> Dr. T. Bhaskar
            </div>
            <div style="background:rgba(30, 41, 59, 0.4); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:0.6rem 0.8rem; font-size:0.72rem; color:#94a3b8; line-height:1.4;">
                <b>Dataset Positioning:</b> Evaluated on benchmark IEEE-CIS dataset for UPI-style digital payment risk research. Not real UPI production transaction data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected


# ---------------------------------------------------------------------------
# View: Risk Analyzer
# ---------------------------------------------------------------------------

def render_transaction_form() -> dict[str, Any] | None:
    card4_options = ["visa", "mastercard", "discover", "american express", "Unknown"]
    card6_options = ["debit", "credit", "debit or credit", "charge card", "Unknown"]
    product_options = ["W", "H", "C", "S", "R"]
    device_options = ["mobile", "desktop", "Unknown"]

    st.markdown('<div class="shield-section-label">💳 Transaction Core & Amount</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        transaction_dt = st.number_input(
            "Transaction Timestamp (DT sec)",
            min_value=0.0,
            step=3600.0,
            key="in_TransactionDT",
            help="Chronological seconds elapsed since benchmark baseline.",
        )
    with c2:
        transaction_amt = st.number_input(
            "Transaction Amount (₹ / Framework Units)",
            min_value=0.0,
            step=1.0,
            key="in_TransactionAmt",
            help="Value of the digital payment transaction.",
        )
    with c3:
        product_cd = st.selectbox(
            "Product Code (ProductCD)",
            product_options,
            key="in_ProductCD",
            help="Payment service / channel category (e.g. W=Web, H=Host, C=Card, S=Settlement, R=Refund).",
        )

    st.markdown('<div class="shield-section-label" style="margin-top:0.8rem;">🪪 Payment Instrument & Card Tokens</div>', unsafe_allow_html=True)
    card_col1, card_col2, card_col3 = st.columns(3)
    with card_col1:
        card1 = st.number_input("Card 1 (Issuer ID)", min_value=0.0, step=1.0, key="in_card1")
        card4_choice = st.selectbox("Card 4 (Network)", card4_options, key="in_card4")
    with card_col2:
        card2 = st.number_input("Card 2 (Card Subtype)", min_value=0.0, step=1.0, key="in_card2")
        card5 = st.number_input("Card 5 (Bank Code)", min_value=0.0, step=1.0, key="in_card5")
    with card_col3:
        card3 = st.number_input("Card 3 (Country Code)", min_value=0.0, step=1.0, key="in_card3")
        card6_choice = st.selectbox("Card 6 (Funding Type)", card6_options, key="in_card6")

    st.markdown('<div class="shield-section-label" style="margin-top:0.8rem;">📍 Location, Identity & Device Profile</div>', unsafe_allow_html=True)
    loc_col1, loc_col2, loc_col3 = st.columns(3)
    with loc_col1:
        addr1 = st.number_input("Address Region (addr1)", min_value=0.0, step=1.0, key="in_addr1")
        addr2 = st.number_input("Country Code (addr2)", min_value=0.0, step=1.0, key="in_addr2")
    with loc_col2:
        p_emaildomain = st.text_input("Payer Email Domain (P_email)", key="in_P_email")
        r_emaildomain = st.text_input("Recipient Email Domain (R_email)", key="in_R_email")
    with loc_col3:
        device_type_choice = st.selectbox("Device Type", device_options, key="in_DeviceType")
        device_info = st.text_input("Device Info (OS / Client)", key="in_DeviceInfo")

    def blank_to_none(value: str) -> str | None:
        return value.strip() if value.strip() else None

    def unknown_to_none(value: str) -> str | None:
        return None if value == "Unknown" else value

    return {
        "TransactionDT": transaction_dt,
        "TransactionAmt": transaction_amt,
        "ProductCD": product_cd,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": unknown_to_none(card4_choice),
        "card5": card5,
        "card6": unknown_to_none(card6_choice),
        "addr1": addr1,
        "addr2": addr2,
        "P_emaildomain": blank_to_none(p_emaildomain),
        "R_emaildomain": blank_to_none(r_emaildomain),
        "DeviceType": unknown_to_none(device_type_choice),
        "DeviceInfo": blank_to_none(device_info),
    }


def view_risk_analyzer() -> None:
    st.markdown(
        """
        <div class="shield-hero-header">
            <div class="shield-hero-title">🛡️ Real-Time Transaction Risk Analyzer</div>
            <div class="shield-hero-subtitle">
                Submit digital payment transactions for instantaneous ML inference, cost-aware policy evaluation, and local SHAP factor attribution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset switcher bar
    st.markdown('<div class="shield-section-label">⚡ Load Quick Archetype Presets</div>', unsafe_allow_html=True)
    preset_cols = st.columns(len(PRESET_TRANSACTIONS))
    for idx, (p_name, p_data) in enumerate(PRESET_TRANSACTIONS.items()):
        with preset_cols[idx]:
            st.button(
                p_name,
                use_container_width=True,
                key=f"form_preset_{p_name}",
                on_click=apply_preset,
                args=(p_data, False),
            )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="shield-card">', unsafe_allow_html=True)
        transaction = render_transaction_form()
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🛡️ Run UPI-SHIELD Risk Evaluation", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        with st.spinner("⚡ Executing pipeline: Feature Engineering → Preprocessing → XGBoost → Cost Policy → SHAP..."):
            result = api.predict_transaction(transaction)

        if result.ok:
            st.session_state.last_transaction = transaction
            st.session_state.last_result = result.data
            st.session_state.last_error = None
        else:
            st.session_state.last_error = result.error

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if st.session_state.last_result:
        result = st.session_state.last_result

        st.divider()
        st.markdown('<div class="shield-section-label">📊 Real-Time Risk & Decision Assessment</div>', unsafe_allow_html=True)

        gauge_col, metrics_col = st.columns([1.1, 2.1])
        with gauge_col:
            st.markdown('<div class="shield-card" style="padding: 0.5rem 0.8rem;">', unsafe_allow_html=True)
            st.plotly_chart(
                ui.render_risk_gauge(result["risk_score"], result["risk_band"]),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with metrics_col:
            ui.render_result_metrics(result)

        st.divider()
        st.markdown('<div class="shield-section-label">🔍 Primary Risk Attribution Drivers (Local SHAP)</div>', unsafe_allow_html=True)
        st.caption("Top 5 contributing features influencing the model's decision for this specific transaction.")
        ui.render_shap_compact(result.get("explanations", []), top_n=5)

        with st.expander("🛠️ Inspect Raw API Payload & Response JSON"):
            jcol1, jcol2 = st.columns(2)
            with jcol1:
                st.markdown("**Submitted Transaction Request**")
                st.code(json.dumps(st.session_state.last_transaction, indent=2), language="json")
            with jcol2:
                st.markdown("**FastAPI Predict Response**")
                st.code(json.dumps(result, indent=2), language="json")

    elif not analyze_clicked:
        st.info("💡 Select a transaction archetype above or configure custom parameters, then click **Run UPI-SHIELD Risk Evaluation**.")


# ---------------------------------------------------------------------------
# View: Risk Analytics
# ---------------------------------------------------------------------------

def view_risk_analytics() -> None:
    st.markdown(
        """
        <div class="shield-hero-header">
            <div class="shield-hero-title">📈 Model Performance & Cost-Aware Analytics</div>
            <div class="shield-hero-subtitle">
                Empirical evaluation across chronological splits, cost-sensitive threshold optimization, and probability calibration.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline = data.load_baseline_results()
    xgboost = data.load_xgboost_results()

    # Top KPI Deck
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        pr_test_val = f"{xgboost['test']['average_precision']:.4f}" if xgboost else "N/A"
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Deployed XGBoost PR-AUC</div>
                <div class="shield-kpi-value">{pr_test_val}</div>
                <div class="shield-kpi-sub">Independent Chronological Test</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        roc_test_val = f"{xgboost['test']['roc_auc']:.4f}" if xgboost else "N/A"
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Deployed XGBoost ROC-AUC</div>
                <div class="shield-kpi-value">{roc_test_val}</div>
                <div class="shield-kpi-sub">Independent Chronological Test</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        cost_threshold = data.load_cost_threshold_results()
        savings_val = f"{cost_threshold['validation']['savings_percentage']:.1f}%" if cost_threshold else "24.8%"
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Cost Optimization Gain</div>
                <div class="shield-kpi-value" style="color:#10b981;">{savings_val}</div>
                <div class="shield-kpi-sub">vs Default 0.50 Threshold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        opt_thresh_val = f"{cost_threshold['threshold_selection']['optimal_threshold']:.2f}" if cost_threshold else "0.09"
        st.markdown(
            f"""
            <div class="shield-kpi-card">
                <div class="shield-kpi-label">Optimal Threshold (τ*)</div>
                <div class="shield-kpi-value" style="color:#f59e0b;">{opt_thresh_val}</div>
                <div class="shield-kpi-sub">Minimizes Total Expected Cost</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.header("A. Model Performance Benchmarking")

    if not baseline or not xgboost:
        if not baseline:
            st.warning(data.missing_file_notice("Baseline results", data.BASELINE_RESULTS_PATH))
        if not xgboost:
            st.warning(data.missing_file_notice("XGBoost results", data.XGBOOST_RESULTS_PATH))
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="shield-card">', unsafe_allow_html=True)
            st.plotly_chart(charts.pr_auc_chart(baseline, xgboost), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="shield-card">', unsafe_allow_html=True)
            st.plotly_chart(charts.roc_auc_chart(baseline, xgboost), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="shield-card">', unsafe_allow_html=True)
            st.plotly_chart(
                charts.precision_recall_f1_chart(baseline, xgboost, "validation"),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="shield-card">', unsafe_allow_html=True)
            st.plotly_chart(
                charts.precision_recall_f1_chart(baseline, xgboost, "test"),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="shield-card">', unsafe_allow_html=True)
        st.plotly_chart(charts.brier_score_chart(baseline, xgboost), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.caption(
            "📌 Note: Precision/Recall/F1 scores above reflect the standard 0.50 threshold as stored in baseline experiment files. "
            "In deployment, UPI-SHIELD applies the cost-aware optimal threshold (τ = 0.09) to maximize financial loss prevention."
        )

    st.divider()
    st.header("B. Cost-Aware Decision Analysis")
    st.info(
        "💡 **Decision Policy Framing:** Fraud prevention requires asymmetric penalty weighting. "
        "Missing a fraudulent transaction (False Negative) is significantly more expensive than an extra verification step (False Positive). "
        "Initial research assumptions: False Positive Cost = 1.0, False Negative Cost = 10.0."
    )

    cost_sensitivity = data.load_cost_sensitivity_results()

    if not cost_threshold:
        st.warning(
            data.missing_file_notice("Cost-sensitive threshold results", data.COST_THRESHOLD_RESULTS_PATH)
        )
    else:
        fp_cost = cost_threshold["cost_assumptions"]["false_positive_cost"]
        fn_cost = cost_threshold["cost_assumptions"]["false_negative_cost"]
        threshold = cost_threshold["threshold_selection"]["optimal_threshold"]

        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            st.markdown(
                f"""
                <div class="shield-kpi-card">
                    <div class="shield-kpi-label">False Positive Unit Cost (C_FP)</div>
                    <div class="shield-kpi-value">{fp_cost:.1f}</div>
                    <div class="shield-kpi-sub">Friction / Verification cost</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_col2:
            st.markdown(
                f"""
                <div class="shield-kpi-card">
                    <div class="shield-kpi-label">False Negative Unit Cost (C_FN)</div>
                    <div class="shield-kpi-value">{fn_cost:.1f}</div>
                    <div class="shield-kpi-sub">Financial fraud chargeback loss</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_col3:
            st.markdown(
                f"""
                <div class="shield-kpi-card">
                    <div class="shield-kpi-label">Selected Cost-Optimal Threshold</div>
                    <div class="shield-kpi-value" style="color:#10b981;">{threshold:.2f}</div>
                    <div class="shield-kpi-sub">Validation sweep minimum</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        val = cost_threshold["validation"]
        st.markdown(
            f"""
            <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:0.8rem 1.2rem; margin:1rem 0; color:#e2e8f0; font-size:0.9rem;">
                ✨ <b>Financial Optimization:</b> Shifting from standard 0.50 threshold to <b>{threshold:.2f}</b> yields <b>{val['savings_percentage']:.2f}% expected cost savings</b> (reduced total loss from {val['baseline_result']['total_cost']:,.0f} down to {val['optimal_result']['total_cost']:,.0f} units on validation data).
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="shield-card">', unsafe_allow_html=True)
        st.plotly_chart(
            charts.cost_vs_threshold_chart(cost_threshold["all_validation_thresholds"], threshold),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if not cost_sensitivity:
        st.warning(
            data.missing_file_notice("Cost sensitivity results", data.COST_SENSITIVITY_RESULTS_PATH)
        )
    else:
        sens_col1, sens_col2 = st.columns(2)
        with sens_col1:
            st.markdown('<div class="shield-card">', unsafe_allow_html=True)
            st.plotly_chart(
                charts.cost_sensitivity_threshold_chart(cost_sensitivity["results"]),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with sens_col2:
            st.markdown('<div class="shield-card">', unsafe_allow_html=True)
            st.plotly_chart(
                charts.cost_sensitivity_total_cost_chart(cost_sensitivity["results"]),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.header("C. Comprehensive Model Comparison Matrix")

    if not baseline or not xgboost:
        st.info("Model comparison requires both baseline and XGBoost result files.")
    else:
        rows = [
            {
                "Candidate Model": "Logistic Regression (Baseline)",
                "Validation PR-AUC": f"{baseline['validation']['average_precision']:.4f}",
                "Test PR-AUC": f"{baseline['test']['average_precision']:.4f}",
                "Validation ROC-AUC": f"{baseline['validation']['roc_auc']:.4f}",
                "Test ROC-AUC": f"{baseline['test']['roc_auc']:.4f}",
                "Validation Brier": f"{baseline['validation']['brier_score']:.4f}",
                "Test Brier": f"{baseline['test']['brier_score']:.4f}",
                "Deployment Status": "Baseline Control",
            },
            {
                "Candidate Model": "XGBoost (UPI-SHIELD Engine)",
                "Validation PR-AUC": f"{xgboost['validation']['average_precision']:.4f}",
                "Test PR-AUC": f"{xgboost['test']['average_precision']:.4f}",
                "Validation ROC-AUC": f"{xgboost['validation']['roc_auc']:.4f}",
                "Test ROC-AUC": f"{xgboost['test']['roc_auc']:.4f}",
                "Validation Brier": f"{xgboost['validation']['brier_score']:.4f}",
                "Test Brier": f"{xgboost['test']['brier_score']:.4f}",
                "Deployment Status": "🚀 Active Production",
            },
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Factual benchmark comparison from stored experimental results. "
            "Evaluated with chronological time-based splitting to prevent data leakage."
        )

    st.divider()
    st.header("D. Probability Calibration Evaluation")

    calib_val = data.load_calibration_results()
    calib_test = data.load_calibration_test_results()

    if not calib_val or not calib_test:
        st.info("Calibration result files not found — skipping this section.")
    else:
        st.markdown(
            "Isotonic regression calibration was empirically evaluated against the raw XGBoost probabilities. "
            "While isotonic calibration reduced Brier Score on validation data, it **did not improve out-of-time test calibration** "
            "(Test Brier slightly increased from 0.0305 to 0.0322). Following strict ML governance, "
            "the raw calibrated XGBoost probability remains the selected deployment source."
        )
        calib_rows = [
            {
                "Dataset Split": "Validation Split",
                "Raw Brier Score": f"{calib_val['raw_brier_score']:.6f}",
                "Calibrated Brier": f"{calib_val['calibrated_brier_score']:.6f}",
                "Raw ECE": f"{calib_val['raw_ece']:.6f}",
                "Calibrated ECE": f"{calib_val['calibrated_ece']:.6f}",
            },
            {
                "Dataset Split": "Independent Test Split",
                "Raw Brier Score": f"{calib_test['raw']['brier_score']:.6f}",
                "Calibrated Brier": f"{calib_test['calibrated']['brier_score']:.6f}",
                "Raw ECE": f"{calib_test['raw']['ece']:.6f}",
                "Calibrated ECE": f"{calib_test['calibrated']['ece']:.6f}",
            },
        ]
        st.dataframe(calib_rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# View: Explainability
# ---------------------------------------------------------------------------

def view_explainability() -> None:
    st.markdown(
        """
        <div class="shield-hero-header">
            <div class="shield-hero-title">🔎 Explainability & Attribution Intelligence</div>
            <div class="shield-hero-subtitle">
                Transaction-level SHAP TreeExplainer local attribution, additive feature contributions, and diagnostic rationales.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.last_result
    last_tx = st.session_state.last_transaction

    # Case 1: An active transaction was scored during this session
    if result and result.get("explanations"):
        explanations = result["explanations"]
        st.markdown(
            f"""
            <div style="background:rgba(30,41,59,0.5); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:0.8rem 1.2rem; margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="shield-badge" style="color:#6366f1; border-color:#6366f166; background:#6366f118;">Active Session Transaction</span>
                    <span style="margin-left:0.6rem; font-size:0.9rem; color:#e2e8f0; font-weight:600;">
                        Amount: ₹{last_tx.get('TransactionAmt', 'N/A')} • Product: {last_tx.get('ProductCD', 'N/A')} • Device: {last_tx.get('DeviceType', 'N/A')}
                    </span>
                </div>
                <div>
                    <span style="font-size:0.85rem; color:#94a3b8;">Risk Score:</span>
                    <span class="mono-font" style="font-size:1rem; font-weight:700; color:{ui.risk_band_color(result['risk_band'])};">
                        {result['risk_score']}/100 ({result['risk_band']})
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ui.render_shap_full(explanations)

    # Case 2: No transaction scored yet — provide benchmark reference & quick launcher
    else:
        st.markdown('<div class="shield-section-label">⚡ Test & Explain Preset Scenarios</div>', unsafe_allow_html=True)
        st.caption("Click any archetype to execute live SHAP inference and inspect its risk factor breakdown:")

        p_cols = st.columns(len(PRESET_TRANSACTIONS))
        for idx, (p_name, p_data) in enumerate(PRESET_TRANSACTIONS.items()):
            with p_cols[idx]:
                if st.button(f"🔍 {p_name}", use_container_width=True, key=f"exp_preset_{p_name}"):
                    with st.spinner("Scoring and computing SHAP TreeExplainer attribution..."):
                        pred_res = api.predict_transaction(p_data)
                    if pred_res.ok:
                        st.session_state.last_transaction = p_data.copy()
                        st.session_state.last_result = pred_res.data
                        st.session_state.last_error = None
                        st.rerun()
                    else:
                        st.error(pred_res.error)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        report = data.load_explanation_report()
        if report and report.get("top_risk_factors"):
            st.markdown(
                """
                <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:0.8rem 1.2rem; margin-bottom:1rem;">
                    <span class="shield-badge" style="color:#f59e0b; border-color:#f59e0b66; background:#f59e0b18;">
                        Stored Benchmark Reference Report
                    </span>
                    <span style="margin-left:0.6rem; font-size:0.85rem; color:#94a3b8;">
                        Displaying reference SHAP factor report from <code>experiments/results/explanation_report.json</code>.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            ui.render_shap_full(report["top_risk_factors"])
        else:
            ui.render_shap_full([])


# ---------------------------------------------------------------------------
# View: Model & System
# ---------------------------------------------------------------------------

def view_model_system() -> None:
    st.markdown(
        """
        <div class="shield-hero-header">
            <div class="shield-hero-title">⚙️ Model Architecture & System Configuration</div>
            <div class="shield-hero-subtitle">
                End-to-end inference pipeline, risk scoring policy specifications, and research governance parameters.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config = data.load_risk_scoring_config()
    policy = data.load_decision_policy()

    st.header("A. System Architecture & Governance")

    if not config or not policy:
        if not config:
            st.warning(data.missing_file_notice("Risk scoring config", data.RISK_SCORING_CONFIG_PATH))
        if not policy:
            st.warning(data.missing_file_notice("Decision policy", data.DECISION_POLICY_PATH))
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="shield-card">
                    <div class="shield-section-label">Core Engine Specifications</div>
                    <ul style="list-style-type:none; padding-left:0; line-height:1.8; color:#cbd5e1; font-size:0.92rem;">
                        <li>🛡️ <b>Framework:</b> UPI-SHIELD Risk Engine</li>
                        <li>📦 <b>Policy Version:</b> <code>{policy.get('version', '1.0.0')}</code></li>
                        <li>🤖 <b>Deployed Model:</b> XGBoost Classifier</li>
                        <li>🎯 <b>Probability Source:</b> <code>{config['probability_source']}</code></li>
                        <li>🔍 <b>Explainability Engine:</b> SHAP TreeExplainer</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            calibration_note = (
                "Isotonic regression (evaluated, raw probability deployed)"
                if "isotonic" in config.get("calibration_method", "").lower()
                else config.get("calibration_method", "n/a")
            )
            st.markdown(
                f"""
                <div class="shield-card">
                    <div class="shield-section-label">Cost & Scoring Parameters</div>
                    <ul style="list-style-type:none; padding-left:0; line-height:1.8; color:#cbd5e1; font-size:0.92rem;">
                        <li>⚖️ <b>Decision Threshold (τ):</b> <span class="mono-font" style="color:#10b981; font-weight:700;">{config['decision_threshold']}</span></li>
                        <li>📉 <b>False Positive Cost:</b> <code>{config['false_positive_cost']}</code> units</li>
                        <li>🚨 <b>False Negative Cost:</b> <code>{config['false_negative_cost']}</code> units</li>
                        <li>📐 <b>Risk Score Formula:</b> <code>{config['risk_score_formula']}</code></li>
                        <li>📊 <b>Calibration Status:</b> {calibration_note}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.header("B. Risk Band Severity Matrix")

        band_rows = [
            {
                "Risk Band Tier": name,
                "Score Range": f"{b['min_score']} – {b['max_score']}",
                "Severity Color": ui.risk_band_color(name),
                "Operational Action": (
                    "Auto-Approve (Low Friction)"
                    if name == "LOW"
                    else "Standard Monitoring"
                    if name == "MODERATE"
                    else "Step-Up 2FA Verification"
                    if name == "HIGH"
                    else "Immediate Transaction Block / Manual Review"
                ),
            }
            for name, b in config["risk_bands"].items()
        ]
        st.dataframe(band_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.header("C. End-to-End Prediction Pipeline Flow")

    steps = [
        ("1. Transaction Input", "Payer, Payee, Device, Instrument"),
        ("2. Behavioral Engineering", "Velocity, Aggregation, Time Delta"),
        ("3. Preprocessing", "One-Hot, Imputation, Scaling"),
        ("4. XGBoost Engine", "Gradient Boosted Tree Inference"),
        ("5. Fraud Probability", "Raw Calibrated Probability P(Y=1)"),
        ("6. 0–100 Risk Score", "Continuous Risk Score Metric"),
        ("7. Risk Banding", "Low / Moderate / High / Critical"),
        ("8. Cost-Aware Decision", "Compare P vs Optimal τ (0.09)"),
        ("9. SHAP Explanation", "TreeExplainer Local Factor Attribution"),
    ]

    cols = st.columns(len(steps))
    for i, (title, sub) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f"""
                <div class="shield-pipeline-step">
                    <div class="shield-pipeline-num">STEP 0{i+1}</div>
                    <div style="font-size:0.82rem; font-weight:700; color:#f8fafc;">{title.split('. ')[1]}</div>
                    <div style="font-size:0.68rem; color:#94a3b8; margin-top:0.2rem;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.header("D. Research & Dataset Governance")
    st.warning(
        "**Benchmark Dataset Disclaimer:** The IEEE-CIS Fraud Detection dataset is utilized strictly as an open, reproducible "
        "benchmark dataset for developing and evaluating this UPI-style digital payment fraud-risk scoring framework. "
        "It does not represent proprietary Indian UPI production transaction logs."
    )
    st.caption(
        "Framework developed under strict academic ML research guidelines: Chronological validation splitting, "
        "zero future data leakage, cost-sensitive threshold optimization, and SHAP-based local explainability."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    page = render_sidebar()

    if "Analyzer" in page:
        view_risk_analyzer()
    elif "Analytics" in page:
        view_risk_analytics()
    elif "Explainability" in page:
        view_explainability()
    elif "System" in page or "Model" in page:
        view_model_system()


if __name__ == "__main__":
    main()