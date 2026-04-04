from decimal import Decimal

from app.core.config import settings


def build_payment_payload(event_title: str, amount: Decimal, booking_token: str) -> str:
    parts = [
        "MANUAL_TRANSFER",
        f"recipient={settings.payment_recipient_name}",
        f"bank={settings.payment_bank_name}",
        f"phone={settings.payment_phone}",
        f"account={settings.payment_account_number}",
        f"bic={settings.payment_bic}",
        f"amount={amount}",
        f"purpose={settings.payment_note_prefix} {event_title} #{booking_token[:8]}",
    ]
    return "\n".join(parts)
