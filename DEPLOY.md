# Deploying the SPY Hedge Cockpit for Free

The Streamlit app (`app.py`) was designed so you can publish it on **Streamlit Community Cloud** and obtain a free, publicly shareable URL. Follow the steps below once you have pushed your fork of this repository to GitHub.

1. **Create a Streamlit Community Cloud account**  
   Go to [streamlit.io](https://streamlit.io) and sign in with GitHub. Authorize the Streamlit app so it can read your repositories.

2. **Ensure your repo contains the app assets**  
   Commit and push the following files:
   - `app.py`
   - the `hedge_app/` package
   - `.streamlit/config.toml`
   - `requirements.txt`

3. **Launch a new Streamlit app**  
   In the Streamlit dashboard choose **“New app”**, then select the repository, branch, and set the “Main file path” to `app.py`.

4. **Set secrets (optional)**  
   No secrets are required. The app fetches SPY and VIX prices from Yahoo Finance at runtime. If you prefer to cache data, you can supply a Yahoo Finance RapidAPI key as an environment variable, but it is not required.

5. **Deploy**  
   Click **“Deploy”**. Streamlit installs the dependencies from `requirements.txt`, runs `app.py`, and provides a public URL such as `https://your-handle-streamlit.app`. You can share this link freely; it remains active while your Streamlit project stays enabled.

6. **Iterate**  
   Every push to the selected branch automatically redeploys the app. Use the Streamlit dashboard to manage app versions, view logs, and restart the service if necessary.

For alternative hosting (HuggingFace Spaces, Render, etc.), the same codebase works because it only requires Python ≥3.9 with the packages listed above.
