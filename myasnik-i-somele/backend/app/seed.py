from app.db.database import SessionLocal
from app.services.seed import seed_all


def main() -> None:
    session = SessionLocal()
    try:
        seed_all(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
