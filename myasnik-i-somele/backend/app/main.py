from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging, logger
from app.db.database import SessionLocal
from app.services.seed import seed_all


configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    session = SessionLocal()
    try:
        seed_all(session)
    finally:
        session.close()
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("request_started method=%s path=%s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("request_finished method=%s path=%s status=%s", request.method, request.url.path, response.status_code)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("unhandled_exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def root_health() -> dict[str, str]:
    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
    finally:
        session.close()
    return {"status": "ok"}


app.include_router(router, prefix=settings.api_prefix)
