# Thesis
SPY Options Risk Management App : https://cvarraj.streamlit.app/

## SPY Hedge Cockpit

This repository now includes a Streamlit front-end (`app.py`) that wraps the CVaR hedge optimizer from the notebook into an interactive, Robinhood-inspired experience. Key features:

- Editable option quote grid with bid/ask support
- Live SPY/VIX quote cards with an intraday hover chart and regime commentary
- Scenario controls for volatility regimes, Monte Carlo sample size, and CVaR tail level
- Portfolio permissions (retail vs. institutional), contract limits, and rounding preferences
- Visualizations for payoff-at-expiry and scenario P&L distributions

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app fetches SPY and VIX history via Yahoo Finance on first load (cached on subsequent runs)
and refreshes intraday quotes every 60 seconds.

