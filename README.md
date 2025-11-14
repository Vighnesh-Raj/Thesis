# Thesis
A Walk-Forward Analysis of Forecasting and Trading Ethereum: From Linear Models to Temporal Fusion Transformers

## SPY Hedge Cockpit (Simplified)

The Streamlit application in this repository keeps the thesis objective—hedging a spot SPY exposure with listed options—while
pairing the workflow down to a single readable module (`app.py`). The app now focuses on the essentials:

- Fetch recent SPY/VIX data for market context and a regime suggestion.
- Let you edit a compact table of SPY option quotes (strike, call/put, mid price).
- Bootstrap price paths from historical returns and solve a linear-program CVaR hedge.
- Display the recommended contract counts and compare unhedged vs. hedged P&L distributions.

### Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads SPY/VIX data from Yahoo Finance and caches it for five minutes. Click **Run optimization** after you
adjust the option board and scenario settings.

### Publishing a shareable link

Deploy the repository to [Streamlit Community Cloud](https://streamlit.io/) to obtain a free, temporary URL. After pushing the
code to GitHub, select "New app" in Streamlit Cloud, point it at `app.py`, and redeploy whenever you commit updates.
