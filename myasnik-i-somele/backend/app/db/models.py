from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class EventStatus(StrEnum):
    draft = "draft"
    published = "published"
    archived = "archived"


class BookingStatus(StrEnum):
    pending_payment = "pending_payment"
    paid = "paid"
    cancelled = "cancelled"
    sold_out = "sold_out"


class PaymentStatus(StrEnum):
    pending = "pending"
    paid = "paid"
    cancelled = "cancelled"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class AdminUser(TimestampMixin, Base):
    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    confirmed_payments: Mapped[list["Payment"]] = relationship(back_populates="confirmed_by_admin")


class Event(TimestampMixin, Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    short_description: Mapped[str] = mapped_column(String(280), nullable=False)
    venue: Mapped[str] = mapped_column(String(255), nullable=False, default="Иркутск, Карла Маркса, 5")
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    price_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="RUB")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=EventStatus.draft.value)
    hero_note: Mapped[str | None] = mapped_column(String(140), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    bookings: Mapped[list["Booking"]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        order_by="Booking.created_at.desc()",
    )


class Booking(TimestampMixin, Base):
    __tablename__ = "bookings"
    __table_args__ = (UniqueConstraint("booking_token", name="uq_bookings_booking_token"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True, nullable=False)
    customer_name: Mapped[str] = mapped_column(String(120), nullable=False)
    phone: Mapped[str] = mapped_column(String(40), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    seats: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=BookingStatus.pending_payment.value)
    payment_status: Mapped[str] = mapped_column(String(32), nullable=False, default=PaymentStatus.pending.value)
    amount_due: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    booking_token: Mapped[str] = mapped_column(String(64), default=lambda: uuid4().hex, nullable=False)
    customer_comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    event: Mapped["Event"] = relationship(back_populates="bookings")
    payment: Mapped["Payment"] = relationship(
        back_populates="booking",
        uselist=False,
        cascade="all, delete-orphan",
    )


class Payment(TimestampMixin, Base):
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(primary_key=True)
    booking_id: Mapped[int] = mapped_column(ForeignKey("bookings.id", ondelete="CASCADE"), unique=True, nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False, default="manual_qr")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=PaymentStatus.pending.value)
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="RUB")
    qr_payload: Mapped[str] = mapped_column(Text, nullable=False)
    recipient_name: Mapped[str] = mapped_column(String(255), nullable=False)
    recipient_bank: Mapped[str] = mapped_column(String(255), nullable=False)
    recipient_phone: Mapped[str] = mapped_column(String(40), nullable=False)
    recipient_account: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recipient_bic: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payment_purpose: Mapped[str] = mapped_column(String(255), nullable=False)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confirmed_by_admin_id: Mapped[int | None] = mapped_column(ForeignKey("admin_users.id"), nullable=True)

    booking: Mapped["Booking"] = relationship(back_populates="payment")
    confirmed_by_admin: Mapped[AdminUser | None] = relationship(back_populates="confirmed_payments")
