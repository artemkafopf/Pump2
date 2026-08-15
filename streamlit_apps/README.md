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

To run the **TTF viewer** — the covariate-binning explorer for the Свод fact panel — pick the x-axis (frequency,
Ql, **Qном** — a discrete axis, one bin per nameplate size with no n-target build and the
marker on the value, Kпод, GLF, water cut, Pзаб, Pзаб/Pнас), movable / mergeable / splittable bins,
mean vs median vs geometric mean with a fitted trend, optional removal of a fitted model's
hazard layers as an AFT time-scale offset, and the censoring-aware KM+RMST panel.  The layer
models are **Пофондовая v5** — one θ_Qном per field (Ya, Vt sour/non-sour, Az, Ic, Au, Mc,
with any unmodelled field falling back to the pooled `Fleet` fit), read from the same two
deploy blocks the calculators are wired from — plus the older single-field Ya v2 / v2.1 / Vt v4.
v5 is the only one offered on «Весь фонд»: every field's θ_Qном is pinned at the same Qном 250,
so a mixed panel is rescaled toward one reference pump instead of several.  The population is any field clearing the failure threshold, the whole fleet,
or the Vt sour / non-sour halves (the v3.2 well-level H₂S class — the strata the Vt models are
fitted on).  A tick next to either chart rescales it to **multiples of a reference bin**
(`Y(ref) = 1`, reference X set in the field beside the tick, median of the covariate by
default), which is how two panels on different levels are compared by shape.  «Отказов в
панели» is a **filtered** count, so «Откуда это число» under the metrics unfolds the whole
selection funnel — sheet rows → resolved outcome → field → closed run with a rate → Мирнинский
cohort → genuine failure → axis coverage and bounds — with what each step dropped.  On Ya that
is 1541 sheet rows down to 630 on the frequency axis, and the largest single cut is the
genuine-failure rule, not any coverage limit:

```bash
streamlit run streamlit_apps/ttf_viewer.py
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

For the frequency-binning explorer:

```bash
.\scripts\setup_streamlit.ps1
.\scripts\run_ttf_viewer.ps1
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
