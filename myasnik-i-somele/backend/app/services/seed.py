from datetime import datetime, timedelta, timezone
from decimal import Decimal

from slugify import slugify
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password
from app.db.models import AdminUser, Event, EventStatus


def ensure_admin(session: Session) -> None:
    admin = session.scalar(select(AdminUser).where(AdminUser.email == settings.admin_email))
    if admin:
        return

    session.add(
        AdminUser(
            email=settings.admin_email,
            name=settings.admin_name,
            password_hash=hash_password(settings.admin_password),
        )
    )
    session.commit()


def ensure_sample_events(session: Session) -> None:
    if not settings.seed_sample_data:
        return

    existing = session.scalar(select(Event.id).limit(1))
    if existing:
        return

    now = datetime.now(timezone.utc)
    events = [
        {
            "title": "Дегустация стейков и красных вин",
            "short_description": "Три подачи мяса, четыре бокала и камерный вечер на 18 гостей.",
            "description": "Фирменный вечер с авторской подачей мяса, подбором вин и коротким рассказом сомелье о сочетаниях.",
            "starts_at": now + timedelta(days=3, hours=11),
            "capacity": 18,
            "price_amount": Decimal("4500.00"),
            "status": EventStatus.published.value,
            "hero_note": "Хит недели",
        },
        {
            "title": "Мясной brunch с игристым",
            "short_description": "Поздний завтрак с тартаром, пастрами и игристым на выходных.",
            "description": "Неспешный субботний бранч с сетом мясных закусок и двумя бокалами игристого.",
            "starts_at": now + timedelta(days=8, hours=8),
            "capacity": 24,
            "price_amount": Decimal("3200.00"),
            "status": EventStatus.published.value,
            "hero_note": "Утренний формат",
        },
        {
            "title": "Школа сомелье: мясо и терруар",
            "short_description": "Образовательный вечер о сортах вин и текстурах мяса.",
            "description": "Практическая встреча для гостей, которые хотят научиться увереннее выбирать вино к блюдам.",
            "starts_at": now + timedelta(days=15, hours=12),
            "capacity": 16,
            "price_amount": Decimal("3900.00"),
            "status": EventStatus.published.value,
            "hero_note": "Для любопытных",
        },
        {
            "title": "Закрытый ужин шефа",
            "short_description": "Небольшой стол на 12 гостей с редкими позициями из винной карты.",
            "description": "Особенный вечер с сет-меню шефа и подбором редких вин по каждому курсу.",
            "starts_at": now + timedelta(days=22, hours=13),
            "capacity": 12,
            "price_amount": Decimal("6900.00"),
            "status": EventStatus.draft.value,
            "hero_note": "Скоро анонс",
        },
    ]

    for item in events:
        session.add(
            Event(
                slug=slugify(item["title"]),
                title=item["title"],
                short_description=item["short_description"],
                description=item["description"],
                venue="Иркутск, Карла Маркса, 5",
                starts_at=item["starts_at"],
                capacity=item["capacity"],
                price_amount=item["price_amount"],
                status=item["status"],
                hero_note=item["hero_note"],
                published_at=item["starts_at"] if item["status"] == EventStatus.published.value else None,
            )
        )

    session.commit()


def seed_all(session: Session) -> None:
    ensure_admin(session)
    ensure_sample_events(session)
