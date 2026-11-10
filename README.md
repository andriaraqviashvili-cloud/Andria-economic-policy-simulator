# Interactive Economic Policy & Inflation Simulator

A Streamlit dashboard that shows how monetary policy, fiscal policy and supply
shocks move inflation, real GDP and unemployment over a 12-quarter horizon.

**Live app:** https://andriaraqviashvili-cloud-andria-economic-policy-simu-app-mlfwnw.streamlit.app

## What it does

Move the sliders in the sidebar and the model recalculates instantly:

- **Monetary policy:** central bank benchmark interest rate (0% to 15%)
- **Fiscal policy:** change in government spending (-20% to +20%) and the tax rate (10% to 50%)
- **External shocks:** an energy / supply shock that adds 3 or 6 percentage points to inflation

The dashboard shows:

- Metric cards for inflation, the output gap, unemployment and GDP growth, each compared with its target or baseline
- A dual-axis forecast of inflation and GDP growth from Q1 to Q12
- An expectations-augmented Phillips Curve showing the shift from the baseline to the new equilibrium
- A downloadable table of the quarterly forecast
- A methodology section with every equation and parameter value

## The model

The engine combines three standard building blocks of intermediate macroeconomics:

| Block | Equation | Role |
|---|---|---|
| IS curve (IS-LM with an interest-rate-setting central bank) | `y* = -α(i - πᵉ - r*) + k_G·s_G·ΔG - β_T(τ - τ₀) - δ·s` | Demand: how interest rates, spending and taxes set the output gap |
| Okun's Law | `u = uₙ - y/θ` | Links the output gap to unemployment |
| Expectations-augmented Phillips Curve | `π = πᵉ + κ·y + s` | Links demand pressure and supply shocks to inflation |

Inflation expectations are partly anchored at the 2% target and partly adaptive
(`πᵉ = λ·π* + (1-λ)·π₋₁`), and output adjusts gradually towards the level demand
is pulling it to. The full derivation is in the app's "Economic Model &
Methodology" section.

## Limitations

This is a stylised teaching model, not a forecasting tool:

- Parameter values are textbook-style illustrations, not estimates for any real economy.
- The policy rate stays fixed over the horizon; the central bank does not react to outcomes.
- The model is linear, so extreme slider settings give proportionally extreme results.

## Run it locally

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Files

| File | Purpose |
|---|---|
| `app.py` | The whole application: model engine, charts and layout |
| `requirements.txt` | Python packages the app needs |
| `.streamlit/config.toml` | Dark theme settings |

## Built with

Python, Streamlit, Plotly, pandas, NumPy
