from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://analysis_user:analysis_password@db:5432/analysis_db"
    cors_origins: str = "http://localhost:5173"
    llm_base_url: str = "http://ollama:11434"
    llm_model: str = "llama3.1:8b"
    llm_timeout_seconds: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


settings = Settings()
