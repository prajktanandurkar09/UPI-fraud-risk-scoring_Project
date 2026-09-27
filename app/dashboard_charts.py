"""
UPI-SHIELD dashboard analytics charts.

Every chart builder here takes already-loaded result dictionaries
(from dashboard_data.py) and returns a Plotly figure. No metric is
computed or estimated in this module — everything plotted is a value
already present in the stored experiment result files.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

MODEL_COLORS = {
    "Logistic Regression": "#38bdf8",  # Sky Blue
    "XGBoost": "#f59e0b",              # Amber / Gold
}

CHART_FONT_FAMILY = "Plus Jakarta Sans, sans-serif"
MONO_FONT_FAMILY = "JetBrains Mono, monospace"


def _grouped_bar(
    categories: list[str],
    series: dict[str, list[float]],
    title: str,
    y_title: str,
) -> go.Figure:
    """Build a modern grouped bar chart with fintech styling."""
    fig = go.Figure()

    for name, values in series.items():
        color = MODEL_COLORS.get(name, "#94a3b8")
        fig.add_trace(
            go.Bar(
                name=name,
                x=categories,
                y=values,
                marker=dict(
                    color=color,
                    line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
                ),
                text=[f"{v:.4f}" for v in values],
                textposition="outside",
                textfont=dict(family=MONO_FONT_FAMILY, size=11, color="#cbd5e1"),
                hovertemplate=f"<b>{name}</b><br>%{{x}}: <b>%{{y:.4f}}</b><extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=CHART_FONT_FAMILY, size=14, color="#f1f5f9"),
        ),
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        yaxis=dict(
            title=dict(text=y_title, font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            tickfont=dict(family=MONO_FONT_FAMILY, size=11, color="#94a3b8"),
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
        ),
        xaxis=dict(
            tickfont=dict(family=CHART_FONT_FAMILY, size=12, color="#e2e8f0"),
            gridcolor="rgba(255, 255, 255, 0.04)",
        ),
        height=380,
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(family=CHART_FONT_FAMILY, size=12, color="#cbd5e1"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def pr_auc_chart(baseline: dict[str, Any], xgboost: dict[str, Any]) -> go.Figure:
    """PR-AUC (Average Precision) comparison chart."""
    return _grouped_bar(
        categories=["Validation", "Test"],
        series={
            "Logistic Regression": [
                baseline["validation"]["average_precision"],
                baseline["test"]["average_precision"],
            ],
            "XGBoost": [
                xgboost["validation"]["average_precision"],
                xgboost["test"]["average_precision"],
            ],
        },
        title="PR-AUC (Average Precision) — Validation vs Test",
        y_title="PR-AUC",
    )


def roc_auc_chart(baseline: dict[str, Any], xgboost: dict[str, Any]) -> go.Figure:
    """ROC-AUC comparison chart."""
    return _grouped_bar(
        categories=["Validation", "Test"],
        series={
            "Logistic Regression": [
                baseline["validation"]["roc_auc"],
                baseline["test"]["roc_auc"],
            ],
            "XGBoost": [
                xgboost["validation"]["roc_auc"],
                xgboost["test"]["roc_auc"],
            ],
        },
        title="ROC-AUC — Validation vs Test",
        y_title="ROC-AUC",
    )


def brier_score_chart(baseline: dict[str, Any], xgboost: dict[str, Any]) -> go.Figure:
    """Brier score (calibration loss) comparison chart."""
    return _grouped_bar(
        categories=["Validation", "Test"],
        series={
            "Logistic Regression": [
                baseline["validation"]["brier_score"],
                baseline["test"]["brier_score"],
            ],
            "XGBoost": [
                xgboost["validation"]["brier_score"],
                xgboost["test"]["brier_score"],
            ],
        },
        title="Brier Score — Validation vs Test (Lower is Better)",
        y_title="Brier Score",
    )


def precision_recall_f1_chart(
    baseline: dict[str, Any], xgboost: dict[str, Any], dataset: str
) -> go.Figure:
    """Precision, Recall, and F1 comparison at default 0.50 threshold."""
    key = dataset.lower()
    return _grouped_bar(
        categories=["Precision", "Recall", "F1"],
        series={
            "Logistic Regression": [
                baseline[key]["precision"],
                baseline[key]["recall"],
                baseline[key]["f1"],
            ],
            "XGBoost": [
                xgboost[key]["precision"],
                xgboost[key]["recall"],
                xgboost[key]["f1"],
            ],
        },
        title=f"Precision / Recall / F1 — {dataset.title()} Set (Default 0.50 Threshold)",
        y_title="Metric Score",
    )


def cost_vs_threshold_chart(
    all_thresholds: list[dict[str, Any]], selected_threshold: float
) -> go.Figure:
    """Total validation cost across the full decision threshold sweep."""

    thresholds = [row["threshold"] for row in all_thresholds]
    costs = [row["total_cost"] for row in all_thresholds]

    fig = go.Figure()

    # Cost curve with subtle gradient fill
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=costs,
            mode="lines",
            line=dict(color="#6366f1", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.08)",
            name="Total Expected Cost",
            hovertemplate="Threshold: <b>%{x:.2f}</b><br>Total Cost: <b>₹%{y:,.0f}</b><extra></extra>",
        )
    )

    selected_row = min(all_thresholds, key=lambda r: abs(r["threshold"] - selected_threshold))

    # Highlight optimal threshold marker
    fig.add_trace(
        go.Scatter(
            x=[selected_row["threshold"]],
            y=[selected_row["total_cost"]],
            mode="markers+text",
            marker=dict(
                color="#ef4444",
                size=14,
                symbol="diamond",
                line=dict(color="#ffffff", width=2),
            ),
            name=f"Optimal τ = {selected_threshold:.2f}",
            text=[f" Optimal τ = {selected_threshold:.2f} (Cost: {selected_row['total_cost']:,.0f})"],
            textposition="top right",
            textfont=dict(family=MONO_FONT_FAMILY, size=11, color="#f87171"),
            hovertemplate=f"<b>Optimal Threshold</b>: {selected_threshold:.2f}<br>Min Cost: ₹{selected_row['total_cost']:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Validation Expected Cost vs Decision Threshold (τ) Sweep</b>",
            font=dict(family=CHART_FONT_FAMILY, size=14, color="#f1f5f9"),
        ),
        xaxis=dict(
            title=dict(text="Decision Threshold (τ)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            tickfont=dict(family=MONO_FONT_FAMILY, size=11, color="#94a3b8"),
            gridcolor="rgba(255, 255, 255, 0.06)",
        ),
        yaxis=dict(
            title=dict(text="Total Expected Cost (Cost Units)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            tickfont=dict(family=MONO_FONT_FAMILY, size=11, color="#94a3b8"),
            gridcolor="rgba(255, 255, 255, 0.06)",
        ),
        height=400,
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(family=CHART_FONT_FAMILY, size=12, color="#cbd5e1"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def cost_sensitivity_threshold_chart(results: list[dict[str, Any]]) -> go.Figure:
    """Relative FN cost vs the resulting selected threshold."""

    labels = [row["cost_ratio"] for row in results]
    thresholds = [row["optimal_threshold"] for row in results]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=thresholds,
            marker=dict(
                color="#6366f1",
                line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
            ),
            text=[f"{t:.2f}" for t in thresholds],
            textposition="outside",
            textfont=dict(family=MONO_FONT_FAMILY, size=11, color="#cbd5e1"),
            hovertemplate="FP:FN Ratio: <b>%{x}</b><br>Optimal Threshold: <b>%{y:.2f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>FP:FN Cost Ratio vs Optimal Decision Threshold (τ*)</b>",
            font=dict(family=CHART_FONT_FAMILY, size=13, color="#f1f5f9"),
        ),
        xaxis=dict(
            title=dict(text="Cost Assumption Ratio (FP : FN)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            tickfont=dict(family=CHART_FONT_FAMILY, size=11, color="#cbd5e1"),
            gridcolor="rgba(255, 255, 255, 0.04)",
        ),
        yaxis=dict(
            title=dict(text="Optimal Threshold (τ*)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            tickfont=dict(family=MONO_FONT_FAMILY, size=11, color="#94a3b8"),
            gridcolor="rgba(255, 255, 255, 0.06)",
        ),
        height=360,
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def cost_sensitivity_total_cost_chart(results: list[dict[str, Any]]) -> go.Figure:
    """Relative FN cost vs the resulting total validation cost."""

    labels = [row["cost_ratio"] for row in results]
    costs = [row["total_cost"] for row in results]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=costs,
            marker=dict(
                color="#f59e0b",
                line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
            ),
            text=[f"{c:,.0f}" for c in costs],
            textposition="outside",
            textfont=dict(family=MONO_FONT_FAMILY, size=11, color="#cbd5e1"),
            hovertemplate="FP:FN Ratio: <b>%{x}</b><br>Total Validation Cost: <b>₹%{y:,.0f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>FP:FN Cost Ratio vs Total Validation Cost</b>",
            font=dict(family=CHART_FONT_FAMILY, size=13, color="#f1f5f9"),
        ),
        xaxis=dict(
            title=dict(text="Cost Assumption Ratio (FP : FN)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            tickfont=dict(family=CHART_FONT_FAMILY, size=11, color="#cbd5e1"),
            gridcolor="rgba(255, 255, 255, 0.04)",
        ),
        yaxis=dict(
            title=dict(text="Total Cost (Framework Units)", font=dict(family=CHART_FONT_FAMILY, size=12, color="#94a3b8")),
            tickfont=dict(family=MONO_FONT_FAMILY, size=11, color="#94a3b8"),
            gridcolor="rgba(255, 255, 255, 0.06)",
        ),
        height=360,
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig