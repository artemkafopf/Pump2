from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Мясник и Сомелье"
    api_prefix: str = "/api"
    environment: str = "development"
    debug: bool = False

    database_url: str = "postgresql+psycopg://mis_user:mis_password@db:5432/mis_db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 720

    admin_email: str = "admin@mis.local"
    admin_password: str = "Admin123!"
    admin_name: str = "Хозяин"

    frontend_url: str = "http://localhost:5173"
    public_base_url: str = "http://localhost:5173"
    timezone_name: str = "Asia/Irkutsk"

    payment_recipient_name: str = "ООО Мясник и Сомелье"
    payment_phone: str = "+79000000000"
    payment_bank_name: str = "СберБанк"
    payment_account_number: str = ""
    payment_bic: str = ""
    payment_note_prefix: str = "Оплата мероприятия"

    seed_sample_data: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @computed_field
    @property
    def cors_origins_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
