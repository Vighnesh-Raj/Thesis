# Thesis
A Walk-Forward Analysis of Forecasting and Trading Ethereum: From Linear Models to Temporal Fusion Transformers

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

#### Troubleshooting: seeing an old UI

If you merged updates but the Streamlit page still looks like the previous version, make sure the
latest files are on disk and that the dependencies are upgraded:

1. Pull the most recent commit: `git pull`
2. Reinstall requirements (new packages such as `plotly` and `streamlit-autorefresh` were added):
   `pip install --upgrade -r requirements.txt`
3. Restart or rerun `streamlit run app.py`. Streamlit caches code aggressively, so a running
   session may need a restart or a browser hard refresh (⌘⇧R / Ctrl+F5) to pick up the new layout.

### Deploy for a free public link

You can publish the Streamlit app on [Streamlit Community Cloud](https://streamlit.io/) and obtain a shareable URL at no cost. See [DEPLOY.md](DEPLOY.md) for a step-by-step guide.
