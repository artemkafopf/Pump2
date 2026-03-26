from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.db.database import Base, engine


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Excel Relationship Analysis API",
    version="1.0.0",
    description="Upload Excel files, keep original data intact, and analyze target relationships with CatBoost.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(router, prefix="/api")

