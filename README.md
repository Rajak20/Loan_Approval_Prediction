# Loan Approval Prediction — Streamlit App

## Files
- `app.py` — Streamlit application
- `loan_model.pkl` — trained Logistic Regression model
- `scaler (3).pkl` — fitted StandardScaler
- `requirements.txt` — Python dependencies

## Run locally

1. Keep all four files in the same folder.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

## Important label assumption
The app currently interprets model output `1` as **Approved** and `0` as **Rejected**.
Confirm this matches the target encoding used when training your model.

## Deployment
Upload `app.py`, both `.pkl` files, and `requirements.txt` to the same GitHub repository.
Deploy the repository using Streamlit Community Cloud and select `app.py` as the entry point.

Use trusted model files only. Pickle files can execute code when loaded.
