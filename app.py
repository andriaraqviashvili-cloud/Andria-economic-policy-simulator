"""
Interactive Economic Policy & Inflation Simulator
=================================================

A Streamlit dashboard that simulates how monetary policy, fiscal policy and
supply shocks move inflation, real GDP and unemployment over a 12-quarter
horizon. The engine is a small dynamic IS-LM (IS-MP) model combined with
Okun's Law and an expectations-augmented Phillips Curve.

Run with:
    pip install streamlit pandas numpy plotly
    streamlit run app.py --theme.base dark

The model is a stylised teaching model with textbook-style calibration. It is
designed to illustrate mechanisms, not to produce real-world forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Economic Policy & Inflation Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit replaced `use_container_width` with `width="stretch"` in late 2025.
# Pick whichever keyword the installed version supports.
_ST_VERSION = tuple(int(part) for part in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 52) else {"use_container_width": True}

# --------------------------------------------------------------------------
# Fixed structural constants (annualised percentages unless stated otherwise)
# --------------------------------------------------------------------------
INFLATION_TARGET = 2.0    # central bank inflation target (%)
NEUTRAL_REAL_RATE = 3.0   # r*: real rate consistent with a zero output gap (%)
BASELINE_POLICY_RATE = INFLATION_TARGET + NEUTRAL_REAL_RATE  # 5.0 %
BASELINE_TAX_RATE = 25.0  # status-quo average tax rate (%)
POTENTIAL_GROWTH = 2.0    # trend real GDP growth (% y/y)

GOV_SHARE_OF_GDP = 0.20   # government purchases as a share of GDP
MPC = 0.60                # marginal propensity to consume
MPM = 0.20                # marginal propensity to import
RATE_SENSITIVITY = 0.45   # alpha: output-gap response to 1pp real-rate gap
TAX_SENSITIVITY = 0.12    # beta_T: output-gap response to 1pp tax-rate change
OUTPUT_PERSISTENCE = 0.55 # rho: share of last quarter's gap that carries over
SHOCK_DEMAND_DRAG = 0.30  # delta: real-income squeeze per 1pp supply shock
SHOCK_PLATEAU_QUARTERS = 4  # quarters the shock stays at full strength
SHOCK_DECAY = 0.60        # per-quarter decay factor after the plateau
UNEMPLOYMENT_FLOOR = 1.0  # frictional lower bound on unemployment (%)
HORIZON = 12              # forecast horizon in quarters

SHOCK_OPTIONS = {
    "None": 0.0,
    "Moderate (+3% Inflation)": 3.0,
    "Severe (+6% Inflation)": 6.0,
}

# Chart palette
COLOR_INFLATION = "#FF9F43"
COLOR_GROWTH = "#4CC9F0"
COLOR_BASELINE = "#8B949E"
COLOR_NEW = "#F472B6"
COLOR_TARGET = "#3FB950"
COLOR_PANEL = "#111827"
COLOR_GRID = "#1F2937"


@dataclass(frozen=True)
class Calibration:
    """Parameters the user can adjust in the sidebar."""

    phillips_slope: float = 0.40   # kappa: inflation response to 1pp output gap
    okun_coefficient: float = 2.0  # output-gap points per 1pp of unemployment
    anchoring: float = 0.75        # lambda: weight of the target in expectations
    natural_unemployment: float = 4.5  # u_n (%)


# --------------------------------------------------------------------------
# Calculation engine
# --------------------------------------------------------------------------
def spending_multiplier(tax_rate: float) -> float:
    """Open-economy Keynesian multiplier with proportional taxes."""
    t = tax_rate / 100.0
    return 1.0 / (1.0 - MPC * (1.0 - t) + MPM)


def supply_shock_path(size: float, horizon: int = HORIZON) -> np.ndarray:
    """Shock holds at full strength for a year, then decays geometrically."""
    quarters = np.arange(1, horizon + 1)
    decay = SHOCK_DECAY ** np.clip(quarters - SHOCK_PLATEAU_QUARTERS, 0, None)
    return size * decay


def simulate(
    policy_rate: float,
    spending_change: float,
    tax_rate: float,
    shock_size: float,
    calibration: Calibration,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Simulate the economy quarter by quarter from the baseline steady state.

    Each quarter:
      1. Expectations:  pi_e = lambda * target + (1 - lambda) * pi[t-1]
      2. IS curve:      y*   = -alpha (i - pi_e - r*) + k_G * s_G * dG
                               - beta_T (tau - tau_0) - delta * shock
      3. Adjustment:    y    = rho * y[t-1] + (1 - rho) * y*
      4. Phillips:      pi   = pi_e + kappa * y + shock
      5. Okun:          u    = u_n - y / okun
      6. Growth (y/y):  g    = g* + (y[t] - y[t-4])
    """
    multiplier = spending_multiplier(tax_rate)
    spending_impulse = multiplier * GOV_SHARE_OF_GDP * spending_change
    tax_impulse = -TAX_SENSITIVITY * (tax_rate - BASELINE_TAX_RATE)
    shocks = supply_shock_path(shock_size, horizon)

    gap_history = [0.0] * 4          # four pre-sample quarters at steady state
    previous_inflation = INFLATION_TARGET
    rows = []

    for t in range(horizon):
        expected_inflation = (
            calibration.anchoring * INFLATION_TARGET
            + (1.0 - calibration.anchoring) * previous_inflation
        )
        real_rate = policy_rate - expected_inflation
        monetary_impulse = -RATE_SENSITIVITY * (real_rate - NEUTRAL_REAL_RATE)

        desired_gap = (
            monetary_impulse
            + spending_impulse
            + tax_impulse
            - SHOCK_DEMAND_DRAG * shocks[t]
        )
        gap = (
            OUTPUT_PERSISTENCE * gap_history[-1]
            + (1.0 - OUTPUT_PERSISTENCE) * desired_gap
        )
        inflation = expected_inflation + calibration.phillips_slope * gap + shocks[t]
        unemployment = max(
            UNEMPLOYMENT_FLOOR,
            calibration.natural_unemployment - gap / calibration.okun_coefficient,
        )
        growth = POTENTIAL_GROWTH + (gap - gap_history[-4])

        rows.append(
            {
                "Quarter": f"Q{t + 1}",
                "Output Gap (%)": gap,
                "Inflation (%)": inflation,
                "Unemployment (%)": unemployment,
                "GDP Growth (% y/y)": growth,
                "Expected Inflation (%)": expected_inflation,
                "Real Interest Rate (%)": real_rate,
                "Supply Shock (pp)": shocks[t],
            }
        )
        gap_history.append(gap)
        previous_inflation = inflation

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def apply_dark_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=17), x=0.01, xanchor="left"),
        paper_bgcolor=COLOR_PANEL,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="Inter, Segoe UI, sans-serif", color="#E6EDF3"),
        margin=dict(l=60, r=60, t=70, b=50),
        height=460,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    return fig


def build_forecast_chart(df: pd.DataFrame, headline_quarter: int) -> go.Figure:
    """Dual-axis time series: inflation (left) vs. GDP growth (right)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["Quarter"], y=df["Inflation (%)"], name="Inflation",
            mode="lines+markers",
            line=dict(color=COLOR_INFLATION, width=3), marker=dict(size=7),
            hovertemplate="%{y:.2f}%<extra>Inflation</extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Quarter"], y=df["GDP Growth (% y/y)"], name="Real GDP growth (y/y)",
            mode="lines+markers",
            line=dict(color=COLOR_GROWTH, width=3, dash="dot"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="%{y:.2f}%<extra>GDP growth</extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Quarter"], y=[INFLATION_TARGET] * len(df),
            name="2.0% inflation target", mode="lines",
            line=dict(color=COLOR_TARGET, width=1.5, dash="dash"),
            hoverinfo="skip",
        ),
        secondary_y=False,
    )
    headline_label = f"Q{headline_quarter}"
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=headline_label, x1=headline_label, y0=0, y1=1,
        line=dict(color=COLOR_BASELINE, width=1, dash="dot"),
    )
    fig.add_annotation(
        x=headline_label, xref="x", y=1.0, yref="paper", yanchor="bottom",
        text="headline", showarrow=False, font=dict(size=11, color=COLOR_BASELINE),
    )

    apply_dark_layout(fig, "12-Quarter Forecast: Inflation vs. GDP Growth")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="Quarter", showgrid=False)
    fig.update_yaxes(
        title_text="Inflation (%)", secondary_y=False,
        title_font_color=COLOR_INFLATION, tickfont_color=COLOR_INFLATION,
        ticksuffix="%",
    )
    fig.update_yaxes(
        title_text="Real GDP growth (% y/y)", secondary_y=True,
        title_font_color=COLOR_GROWTH, tickfont_color=COLOR_GROWTH,
        ticksuffix="%", showgrid=False,
    )
    return fig


def build_phillips_chart(
    df: pd.DataFrame, calibration: Calibration, headline_quarter: int
) -> go.Figure:
    """Short-run Phillips curves (baseline vs. new) and the adjustment path."""
    u_n = calibration.natural_unemployment
    slope = calibration.phillips_slope * calibration.okun_coefficient
    row = df.iloc[headline_quarter - 1]

    # Vertical intercept of each short-run curve at u = u_n
    baseline_intercept = INFLATION_TARGET
    new_intercept = row["Expected Inflation (%)"] + row["Supply Shock (pp)"]

    path_u = np.concatenate([[u_n], df["Unemployment (%)"].to_numpy()])
    path_pi = np.concatenate([[INFLATION_TARGET], df["Inflation (%)"].to_numpy()])
    path_labels = ["Baseline"] + df["Quarter"].tolist()

    u_low = max(0.5, min(path_u.min(), u_n - 3.0) - 0.5)
    u_high = max(path_u.max(), u_n + 3.0) + 0.5
    u_grid = np.linspace(u_low, u_high, 100)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=u_grid, y=baseline_intercept - slope * (u_grid - u_n),
            name="Baseline short-run Phillips curve",
            line=dict(color=COLOR_BASELINE, width=2.5),
            hovertemplate="u = %{x:.2f}%<br>π = %{y:.2f}%<extra>Baseline SRPC</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=u_grid, y=new_intercept - slope * (u_grid - u_n),
            name=f"New short-run Phillips curve (Q{headline_quarter})",
            line=dict(color=COLOR_NEW, width=2.5),
            hovertemplate="u = %{x:.2f}%<br>π = %{y:.2f}%<extra>New SRPC</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path_u, y=path_pi, name="Adjustment path (Q1–Q12)",
            mode="lines+markers", text=path_labels,
            line=dict(color=COLOR_GROWTH, width=1.5, dash="dot"),
            marker=dict(size=5),
            hovertemplate="%{text}<br>u = %{x:.2f}%<br>π = %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[u_n], y=[INFLATION_TARGET], name="Baseline equilibrium",
            mode="markers",
            marker=dict(size=14, color=COLOR_BASELINE, line=dict(color="white", width=1.5)),
            hovertemplate="u = %{x:.2f}%<br>π = %{y:.2f}%<extra>Baseline</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[row["Unemployment (%)"]], y=[row["Inflation (%)"]],
            name=f"New policy equilibrium (Q{headline_quarter})",
            mode="markers",
            marker=dict(size=17, symbol="star", color=COLOR_NEW,
                        line=dict(color="white", width=1.5)),
            hovertemplate="u = %{x:.2f}%<br>π = %{y:.2f}%<extra>New equilibrium</extra>",
        )
    )
    fig.add_vline(
        x=u_n, line=dict(color=COLOR_TARGET, width=1.5, dash="dash"),
        annotation_text="Long-run Phillips curve (uₙ)", annotation_position="top right",
        annotation_font_color=COLOR_TARGET,
    )

    apply_dark_layout(fig, "Expectations-Augmented Phillips Curve")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=70, b=120),
        height=520,
    )
    fig.update_xaxes(title_text="Unemployment rate (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Inflation rate (%)", ticksuffix="%")
    return fig


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0B0F17; color: #E6EDF3; }
        [data-testid="stHeader"] { background: rgba(11, 15, 23, 0.85); }
        [data-testid="stSidebar"] {
            background-color: #0F1623; border-right: 1px solid #1F2937;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label,
        .stApp li { color: #E6EDF3; }
        [data-testid="stMetric"] {
            background: linear-gradient(160deg, #131C2B 0%, #0F1623 100%);
            border: 1px solid #1F2937; border-radius: 12px;
            padding: 16px 18px;
        }
        [data-testid="stMetricLabel"] p {
            color: #9CA3AF; font-size: 0.82rem; letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] { color: #F9FAFB; font-weight: 650; }
        .stance-box {
            background: #111827; border: 1px solid #1F2937; border-radius: 12px;
            padding: 12px 18px; margin: 6px 0 14px 0; color: #C9D1D9;
            font-size: 0.93rem; line-height: 1.6;
        }
        .stance-box b { color: #F9FAFB; }
        .block-container { padding-top: 2.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def deviation_color(deviation: float, higher_is_bad: bool, tolerance: float = 0.05) -> str:
    """Pick st.metric's delta_color so that 'bad' deviations render red."""
    if abs(deviation) < tolerance:
        return "off"
    return "inverse" if higher_is_bad else "normal"


def inflation_delta_color(deviation: float, tolerance: float = 0.05) -> str:
    """Any miss of the inflation target, above or below, is shown in red."""
    if abs(deviation) < tolerance:
        return "off"
    return "inverse" if deviation > 0 else "normal"


def describe_stance(real_rate_gap: float) -> str:
    if real_rate_gap > 0.25:
        return "Restrictive"
    if real_rate_gap < -0.25:
        return "Accommodative"
    return "Neutral"


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main() -> None:
    inject_css()

    # ---------------- Sidebar controls ----------------
    with st.sidebar:
        st.header("Policy Controls")

        st.subheader("Monetary Policy")
        policy_rate = st.slider(
            "Central Bank Benchmark Interest Rate (%)",
            min_value=0.0, max_value=15.0, value=BASELINE_POLICY_RATE, step=0.25,
            format="%.2f%%",
            help="Nominal policy rate. The neutral rate in this model is 5.0% "
                 "(3.0% neutral real rate + 2.0% inflation target).",
        )

        st.subheader("Fiscal Policy")
        spending_change = st.slider(
            "Government Spending Change (%)",
            min_value=-20.0, max_value=20.0, value=0.0, step=1.0, format="%+.0f%%",
            help="Percentage change in real government purchases relative to baseline.",
        )
        tax_rate = st.slider(
            "Tax Rate (%)",
            min_value=10.0, max_value=50.0, value=BASELINE_TAX_RATE, step=1.0,
            format="%.0f%%",
            help="Average tax rate on income. The baseline is 25%.",
        )

        st.subheader("External Shocks")
        shock_label = st.radio(
            "Energy / Supply Shock", options=list(SHOCK_OPTIONS.keys()), index=0,
            help="A cost-push shock that adds directly to inflation for four "
                 "quarters, then fades.",
        )
        shock_size = SHOCK_OPTIONS[shock_label]

        st.divider()
        headline_quarter = st.slider(
            "Headline horizon (quarter)", min_value=1, max_value=HORIZON, value=4,
            help="Quarter reported in the metric cards and used as the new "
                 "equilibrium point on the Phillips Curve chart.",
        )

        with st.expander("Advanced calibration"):
            phillips_slope = st.slider(
                "Phillips Curve slope (κ)", 0.10, 1.00, 0.40, 0.05,
                help="Inflation response (pp) to a 1pp output gap.",
            )
            okun_coefficient = st.slider(
                "Okun coefficient", 1.0, 3.0, 2.0, 0.1,
                help="Output-gap points associated with 1pp of cyclical unemployment.",
            )
            anchoring = st.slider(
                "Expectations anchoring (λ)", 0.0, 1.0, 0.75, 0.05,
                help="1 = expectations fixed at the 2% target; 0 = fully adaptive.",
            )
            natural_unemployment = st.slider(
                "Natural unemployment rate (%)", 3.0, 8.0, 4.5, 0.1,
            )

        st.caption(
            "Stylised teaching model. Results illustrate mechanisms and are "
            "not real-world forecasts."
        )

    calibration = Calibration(
        phillips_slope=phillips_slope,
        okun_coefficient=okun_coefficient,
        anchoring=anchoring,
        natural_unemployment=natural_unemployment,
    )

    # ---------------- Run the model ----------------
    df = simulate(policy_rate, spending_change, tax_rate, shock_size, calibration)
    headline = df.iloc[headline_quarter - 1]

    output_gap = float(headline["Output Gap (%)"])
    inflation = float(headline["Inflation (%)"])
    unemployment = float(headline["Unemployment (%)"])
    growth = float(headline["GDP Growth (% y/y)"])

    inflation_dev = inflation - INFLATION_TARGET
    unemployment_dev = unemployment - calibration.natural_unemployment
    growth_dev = growth - POTENTIAL_GROWTH

    # ---------------- Header ----------------
    st.title("Interactive Economic Policy & Inflation Simulator")
    st.caption(
        "A dynamic IS-LM / expectations-augmented Phillips Curve model. Adjust "
        "the policy levers in the sidebar to see how inflation, output and "
        "unemployment respond over the next 12 quarters."
    )

    # ---------------- Metric cards ----------------
    st.markdown(f"#### Projected outcomes at Q{headline_quarter}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Inflation Rate",
        f"{inflation:.2f}%",
        delta=f"{inflation_dev:+.2f} pp vs 2.0% target",
        delta_color=inflation_delta_color(inflation_dev),
    )
    col2.metric(
        "Real GDP Output Gap",
        f"{output_gap:+.2f}%",
        delta=f"{output_gap:+.2f} pp vs potential",
        delta_color=deviation_color(output_gap, higher_is_bad=False),
    )
    col3.metric(
        "Unemployment Rate",
        f"{unemployment:.2f}%",
        delta=f"{unemployment_dev:+.2f} pp vs natural rate",
        delta_color=deviation_color(unemployment_dev, higher_is_bad=True),
    )
    col4.metric(
        "Real GDP Growth (y/y)",
        f"{growth:.2f}%",
        delta=f"{growth_dev:+.2f} pp vs {POTENTIAL_GROWTH:.1f}% trend",
        delta_color=deviation_color(growth_dev, higher_is_bad=False),
    )

    # ---------------- Policy stance summary ----------------
    multiplier = spending_multiplier(tax_rate)
    real_rate_gap = (policy_rate - INFLATION_TARGET) - NEUTRAL_REAL_RATE
    spending_impulse = multiplier * GOV_SHARE_OF_GDP * spending_change
    tax_impulse = -TAX_SENSITIVITY * (tax_rate - BASELINE_TAX_RATE)
    fiscal_impulse = spending_impulse + tax_impulse
    if fiscal_impulse > 0.05:
        fiscal_stance = "Expansionary"
    elif fiscal_impulse < -0.05:
        fiscal_stance = "Contractionary"
    else:
        fiscal_stance = "Neutral"

    st.markdown(
        f"""
        <div class="stance-box">
        <b>Monetary stance:</b> {describe_stance(real_rate_gap)}
        (real rate {policy_rate - INFLATION_TARGET:.2f}% vs neutral
        {NEUTRAL_REAL_RATE:.2f}%) &nbsp;|&nbsp;
        <b>Fiscal stance:</b> {fiscal_stance}
        (demand impulse {fiscal_impulse:+.2f}% of GDP; spending multiplier
        {multiplier:.2f}) &nbsp;|&nbsp;
        <b>Supply shock:</b> {shock_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Charts ----------------
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_forecast_chart(df, headline_quarter), **STRETCH)
    with right:
        st.plotly_chart(
            build_phillips_chart(df, calibration, headline_quarter), **STRETCH
        )

    # ---------------- Data table ----------------
    with st.expander("Quarterly forecast data"):
        st.dataframe(df.set_index("Quarter").round(2), **STRETCH)
        st.download_button(
            "Download forecast as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="policy_simulation.csv",
            mime="text/csv",
        )

    # ---------------- Methodology ----------------
    with st.expander("Economic Model & Methodology"):
        st.markdown(
            """
This simulator combines three building blocks of intermediate macroeconomics.
All variables are annualised percentages, and the economy starts at its
baseline steady state: a 5.0% policy rate, a 25% tax rate, a zero output gap,
2.0% inflation and unemployment at its natural rate.

##### 1. Aggregate demand: the IS-LM block

The **IS curve** describes goods-market equilibrium. Higher real interest
rates reduce investment and consumption; government spending and lower taxes
raise demand:
"""
        )
        st.latex(
            r"y_t^{*} = -\alpha\,(i - \pi_t^{e} - r^{*}) \;+\; k_G\, s_G\, \Delta G"
            r" \;-\; \beta_T\,(\tau - \tau_0) \;-\; \delta\, s_t"
        )
        st.markdown(
            """
where $y^{*}$ is the output gap demand is pulling towards, $i$ is the policy
rate, $\\pi^{e}$ is expected inflation, $r^{*}$ is the neutral real rate,
$\\Delta G$ is the percentage change in government spending, $s_G$ is the
government share of GDP, $\\tau$ is the tax rate and $s_t$ is the supply
shock (which also squeezes real incomes).

The government spending multiplier is the open-economy Keynesian multiplier
with proportional taxes, so a higher tax rate also weakens fiscal stimulus:
"""
        )
        st.latex(r"k_G = \frac{1}{1 - c\,(1 - \tau) + m}")
        st.markdown(
            """
The **LM curve** describes money-market equilibrium, $M/P = L(Y, i)$. Modern
central banks set the interest rate and supply whatever money is demanded at
that rate, so the LM curve is horizontal at the chosen policy rate (the
"IS-MP" version of IS-LM). The policy-rate slider therefore shifts the LM
curve directly, and equilibrium output is read off the IS curve.

Output adjusts gradually, because spending and investment plans take time to
change:
"""
        )
        st.latex(r"y_t = \rho\, y_{t-1} + (1 - \rho)\, y_t^{*}")

        st.markdown("##### 2. Okun's Law")
        st.markdown(
            "Okun's Law links the output gap to cyclical unemployment. When "
            "output falls below potential, unemployment rises above its "
            "natural rate $u_n$:"
        )
        st.latex(r"u_t = u_n - \frac{y_t}{\theta}")

        st.markdown("##### 3. Expectations-augmented Phillips Curve")
        st.markdown(
            "Inflation depends on expected inflation, demand pressure and "
            "cost-push shocks:"
        )
        st.latex(r"\pi_t = \pi_t^{e} + \kappa\, y_t + s_t")
        st.markdown(
            "Expectations are partly anchored at the central bank's target "
            "$\\pi^{T}$ and partly adaptive:"
        )
        st.latex(r"\pi_t^{e} = \lambda\, \pi^{T} + (1 - \lambda)\, \pi_{t-1}")
        st.markdown(
            """
Substituting Okun's Law gives the curve drawn in the chart,
$\\pi_t = \\pi_t^{e} + s_t - \\kappa\\theta\\,(u_t - u_n)$. Demand policy moves
the economy **along** the short-run curve, while supply shocks and changes in
expected inflation **shift** the whole curve. The long-run Phillips curve is
vertical at $u_n$: there is no permanent trade-off between inflation and
unemployment.

##### 4. GDP growth

Year-on-year real GDP growth equals trend growth plus the change in the
output gap over the previous four quarters:
"""
        )
        st.latex(r"g_t = g^{*} + (y_t - y_{t-4})")

        st.markdown("##### Calibration")
        calibration_table = pd.DataFrame(
            [
                ("Inflation target", "π^T", f"{INFLATION_TARGET:.1f}%"),
                ("Neutral real interest rate", "r*", f"{NEUTRAL_REAL_RATE:.1f}%"),
                ("Trend GDP growth", "g*", f"{POTENTIAL_GROWTH:.1f}%"),
                ("Natural unemployment rate", "u_n",
                 f"{calibration.natural_unemployment:.1f}%"),
                ("Interest sensitivity of demand", "α", f"{RATE_SENSITIVITY:.2f}"),
                ("Marginal propensity to consume", "c", f"{MPC:.2f}"),
                ("Marginal propensity to import", "m", f"{MPM:.2f}"),
                ("Government share of GDP", "s_G", f"{GOV_SHARE_OF_GDP:.2f}"),
                ("Tax-rate sensitivity of demand", "β_T", f"{TAX_SENSITIVITY:.2f}"),
                ("Output persistence", "ρ", f"{OUTPUT_PERSISTENCE:.2f}"),
                ("Supply-shock demand drag", "δ", f"{SHOCK_DEMAND_DRAG:.2f}"),
                ("Phillips Curve slope", "κ", f"{calibration.phillips_slope:.2f}"),
                ("Okun coefficient", "θ", f"{calibration.okun_coefficient:.1f}"),
                ("Expectations anchoring", "λ", f"{calibration.anchoring:.2f}"),
            ],
            columns=["Parameter", "Symbol", "Value"],
        )
        st.dataframe(calibration_table, hide_index=True, **STRETCH)

        st.markdown(
            f"""
##### Assumptions and limitations

- The supply shock adds its full size to inflation for
  {SHOCK_PLATEAU_QUARTERS} quarters and then decays by
  {int((1 - SHOCK_DECAY) * 100)}% per quarter.
- The policy rate is held fixed over the horizon; the central bank does not
  react to the outcomes (no Taylor rule).
- Unemployment cannot fall below a frictional floor of
  {UNEMPLOYMENT_FLOOR:.1f}%. At that floor the economy sits slightly off the
  drawn Phillips curve.
- The model is linear, so extreme slider combinations produce
  proportionally extreme results. Real economies have non-linearities
  (for example the zero lower bound and capacity limits) that are not
  modelled here.
- Parameter values are textbook-style illustrations, not econometric
  estimates for any particular country.
"""
        )


if __name__ == "__main__":
    main()
