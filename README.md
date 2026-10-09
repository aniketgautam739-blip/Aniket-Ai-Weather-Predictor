# SmartGrid AI — Interactive Electricity Forecasting

A student project demonstrating one-hour-ahead electricity-consumption forecasting using a Random Forest model and an interactive Streamlit website.

## Features
- Upload your own hourly CSV data.
- Use a clearly labelled simulated demo dataset if you do not have real data yet.
- Train a Random Forest model in the app.
- Evaluate it on a later-in-time test split and compare against a previous-hour baseline.
- Change recent consumption inputs and forecast hour/day to run a new what-if prediction.
- View history, feature importance, and download cleaned data.

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud
Push `app.py`, `requirements.txt`, and `README.md` to the root of a GitHub repository. On Streamlit Community Cloud, choose that repository, branch `main`, and main file path `app.py`.

## CSV format
Your CSV needs:
- one timestamp column (for example `timestamp`)
- one numeric consumption column (for example `consumption_kwh`, `load_mw`, or `demand`)

Use regularly spaced hourly observations. The app averages duplicate observations within the same hour. Aim for several weeks or months of data if possible. With only a few days of data, model evaluation will be less reliable.

## Important limitation
If no CSV is uploaded, the website uses simulated data so the controls and ML pipeline can be demonstrated. Do not describe simulated demo results as measurements from a real electricity grid. Forecast values use the same units as the dataset.
