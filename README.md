# Excel Relationship Analysis

Full-stack project for uploading Excel files, storing original tables in PostgreSQL, and analyzing relationships with a target variable using CatBoost.

It now also includes an interpretable Weibull stress-reliability workflow:

- reusable fitting code in `backend/analysis`
- a standalone Streamlit UI in `streamlit_apps/` as an alternative to the Vue frontend

## Stack

- Backend: FastAPI
- Frontend: Vue 3 + Vite
- Alternative UI: Streamlit
- Database: PostgreSQL
- Analytics: pandas + CatBoost
- Deployment: Docker Compose

## Start

1. Copy `.env.example` to `.env`
2. Run:

```bash
docker compose up --build
```

3. Open:

- Frontend: `http://localhost:5174`
- Streamlit: `http://localhost:8501`
- Backend docs: `http://localhost:8001/docs`

## Weibull Model

Run the alternative Streamlit app locally:

```bash
.\scripts\setup_streamlit.ps1
.\scripts\run_streamlit.ps1
```

Core Weibull modules live in `backend/analysis`:

- `data_utils.py`: Excel loading, filtering, validation, grouping fallback
- `stress_transforms.py`: stress transform library
- `weibull_model.py`: censored Weibull likelihood fitting with optional stress terms
- `plotting.py`: Plotly chart builders for results and sensitivity analysis
