from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.routes import router
from app.core.config import settings
from app.db.database import Base, engine


Base.metadata.create_all(bind=engine)


def ensure_dataset_storage_fields() -> None:
    with engine.begin() as connection:
        inspector = inspect(connection)
        tables = inspector.get_table_names()
        if "datasets" not in tables:
            return

        existing_columns = {column["name"] for column in inspector.get_columns("datasets")}
        if "storage_section" not in existing_columns:
            connection.execute(text("ALTER TABLE datasets ADD COLUMN storage_section VARCHAR(64)"))
        if "storage_version" not in existing_columns:
            connection.execute(text("ALTER TABLE datasets ADD COLUMN storage_version INTEGER"))

        connection.execute(
            text("UPDATE datasets SET storage_section = 'fact_epu' WHERE storage_section IS NULL OR storage_section = ''")
        )
        connection.execute(
            text(
                """
                WITH ranked AS (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY COALESCE(storage_section, 'fact_epu')
                            ORDER BY created_at, id
                        ) AS version_number
                    FROM datasets
                )
                UPDATE datasets AS d
                SET storage_version = ranked.version_number
                FROM ranked
                WHERE d.id = ranked.id
                  AND (d.storage_version IS NULL OR d.storage_version = 0)
                """
            )
        )


ensure_dataset_storage_fields()

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
