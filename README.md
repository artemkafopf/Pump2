# Excel Relationship Analysis

Full-stack project for uploading Excel files, storing original tables in PostgreSQL, and analyzing relationships with a target variable using CatBoost.

## Stack

- Backend: FastAPI
- Frontend: Vue 3 + Vite
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

- Frontend: `http://localhost:5173`
- Backend docs: `http://localhost:8000/docs`

