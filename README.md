# Energy Price & Generation Forecasting

Forecasting day-ahead electricity prices and renewable generation on the Italian and European power market, using publicly available data from the ENTSO-E Transparency Platform.

> **Status:** Work in progress — data collection and exploratory analysis phase.

---

## Motivation

Electricity markets are among the most complex time series forecasting problems: prices are shaped by physical constraints (grid balance, ramping limits), stochastic inputs (wind, solar), and human behaviour (demand patterns, trading). Accurate short-term forecasts have direct economic value — for grid operators, energy traders, and storage asset managers.

This project applies statistical and machine learning methods to this problem, with a focus on **probabilistic forecasting** (not just point estimates) and **physical interpretability** of model outputs. The latter is informed by a background in nuclear and computational physics, where understanding *why* a model fails is as important as optimising its metrics.

---

## Objectives

- Build a reproducible end-to-end forecasting pipeline for day-ahead electricity prices (EUR/MWh) and generation by source (solar, wind, hydro, gas...)
- Compare classical time series models (SARIMA) against gradient boosting approaches (LightGBM/XGBoost) using rigorous walk-forward cross-validation
- Quantify forecast uncertainty via conformal prediction intervals
- Simulate a simple battery storage dispatch strategy driven by price forecasts, to demonstrate downstream decision-making value

---

## Data

All data is sourced from the **[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)** — the official European Network of Transmission System Operators, freely accessible via API.

| Dataset | Resolution | Coverage |
|---|---|---|
| Day-ahead prices | Hourly | IT, 2022–2024 |
| Actual generation by source | 15-min / hourly | IT, 2022–2024 |
| Cross-border flows (planned) | Hourly | IT borders |

Raw data is not committed to the repository. It can be reproduced by running `src/data_loader.py` with a valid ENTSO-E API key (free registration at transparency.entsoe.eu).

---

## Methods (planned)

**Baseline**
- Naive seasonal forecast (same hour, previous week)
- SARIMA with automatic order selection

**Machine learning**
- LightGBM with lag features, rolling statistics, and calendar variables
- Feature importance analysis for physical interpretation

**Uncertainty quantification**
- Quantile regression
- Conformal prediction intervals (MAPIE)

**Downstream application**
- Rule-based battery dispatch optimisation using price forecasts
- Revenue comparison vs naive strategy

---

## Stack

| Purpose | Library |
|---|---|
| Data access | `entsoe-py` |
| Data manipulation | `pandas`, `numpy` |
| Time series models | `statsmodels` |
| ML models | `lightgbm`, `scikit-learn` |
| Uncertainty | `MAPIE` |
| Visualisation | `plotly`, `matplotlib` |
| Demo (planned) | `streamlit` |

---

## Setup

```bash
git clone https://github.com/your-username/energy-forecasting.git
cd energy-forecasting
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Write your ENTSO-E API inside `.env`.

Then download the data:

```bash
python src/data_loader.py
```

---

## Background

This project is part of a broader effort to apply rigorous quantitative methods — developed through research in experimental nuclear physics (ALICE/CERN collaboration) — to problems in the energy sector. The focus is on methods that are not only accurate but physically interpretable and honest about their uncertainty.

---

## Roadmap

- [x] Repository structure and environment setup
- [x] ENTSO-E data pipeline
- [ ] Exploratory data analysis
- [ ] Baseline models (SARIMA, naive)
- [ ] ML models with feature engineering
- [ ] Probabilistic forecasting
- [ ] Storage dispatch simulation
- [ ] Interactive Streamlit demo

---
