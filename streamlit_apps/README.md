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

To run the latent Weibull and competing-risks playground:

```bash
streamlit run streamlit_apps/latent_weibull_competing_risks.py
```

To run the Bayesian latent Weibull fitter:

```bash
streamlit run streamlit_apps/bayesian_latent_weibull.py
```

To run the Vt short-mode interaction surface explorer:

```bash
streamlit run streamlit_apps/vt_mode_interaction_surface.py
```

The Bayesian app supports both:

- fixed `K` fitting;
- unknown `K` inference with birth/death MCMC and a truncated Poisson prior.

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

For the latent Weibull and competing-risks playground:

```bash
.\scripts\setup_streamlit.ps1
.\scripts\run_latent_weibull_streamlit.ps1
```

For the Bayesian latent Weibull fitter:

```bash
.\scripts\setup_streamlit.ps1
.\scripts\run_bayesian_latent_weibull_streamlit.ps1
```

For the Vt short-mode interaction surface explorer:

```bash
.\scripts\setup_streamlit.ps1
.\scripts\run_vt_mode_interaction_surface_streamlit.ps1
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

Or:

```bash
$env:PYTHONPATH = "backend"
python -m pip install -r streamlit_apps/requirements.txt
python -m streamlit run streamlit_apps/latent_weibull_competing_risks.py
```

Or:

```bash
$env:PYTHONPATH = "backend"
python -m pip install -r streamlit_apps/requirements.txt
python -m streamlit run streamlit_apps/bayesian_latent_weibull.py
```

Or:

```bash
$env:PYTHONPATH = "backend"
python -m pip install -r streamlit_apps/requirements.txt
python -m streamlit run streamlit_apps/vt_mode_interaction_surface.py
```
