# Streamlit Weibull Apps

Alternative UIs for the Weibull stress-reliability workflow.

This folder now also includes a dedicated local CatBoost analysis app that mirrors the original CatBoost dashboard workflow without needing the Vue frontend.

## Run locally

```bash
pip install -r streamlit_apps/requirements.txt
streamlit run streamlit_apps/streamlit_app.py
```

The generic app imports the reusable model code from `backend/analysis`.

To run the new presentation-oriented automated workbook inspector:

```bash
streamlit run streamlit_apps/presentation_app.py
```

To run the CatBoost app locally:

```bash
streamlit run streamlit_apps/catboost_app.py
```

## Recommended local setup

From the repository root:

```bash
.\scripts\setup_streamlit.ps1
.\scripts\run_streamlit.ps1
```

For the presentation app:

```bash
.\scripts\setup_streamlit.ps1
.\scripts\run_presentation_streamlit.ps1
```

For the CatBoost app:

```bash
.\scripts\setup_streamlit.ps1
.\scripts\run_catboost_streamlit.ps1
```

If you prefer not to use the helper script:

```bash
$env:PYTHONPATH = "backend"
python -m pip install -r streamlit_apps/requirements.txt
python -m streamlit run streamlit_apps/streamlit_app.py
```

Or:

```bash
$env:PYTHONPATH = "backend"
python -m pip install -r streamlit_apps/requirements.txt
python -m streamlit run streamlit_apps/presentation_app.py
```

Or:

```bash
$env:PYTHONPATH = "backend"
python -m pip install -r streamlit_apps/requirements.txt
python -m streamlit run streamlit_apps/catboost_app.py
```
