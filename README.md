# customer-churn

Telco customer churn analysis and a **Streamlit** app (`app.py`) that loads `artifacts/logreg_tuned_model.pkl` and `artifacts/preprocessor.pkl`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

[Streamlit Community Cloud](https://streamlit.io/cloud) runs your app from GitHub.

1. **Push this repo to GitHub**  
   Make sure these are committed (the app will not start without them):

   - `app.py`
   - `requirements.txt`
   - `artifacts/logreg_tuned_model.pkl`
   - `artifacts/preprocessor.pkl`

2. **Open** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. **New app** → choose the repository and branch.

4. **Main file path:** `app.py`  
   Leave the default command as `streamlit run` (Cloud fills in the file).

5. **Deploy**  
   Wait for the first build (installing `requirements.txt` can take a few minutes).

6. **If the build fails**, open **App settings → Advanced** and try **Python 3.11**.  
   The project pins `scikit-learn==1.6.1` to match the saved model files; that version is supported on recent Cloud runtimes.

No API keys or **Secrets** are required for this app.

### Optional: pin Python on Cloud

If your platform supports it, you can add a `runtime.txt` in the repo root (check [Streamlit Cloud docs](https://docs.streamlit.io/streamlit-community-cloud) for the current format).
