from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload
from slugify import slugify

from app.api.deps import get_current_admin
from app.core.config import settings
from app.core.security import create_access_token, verify_password
from app.db.database import get_db
from app.db.models import AdminUser, Booking, BookingStatus, Event, EventStatus, Payment, PaymentStatus
from app.schemas.auth import LoginRequest, TokenResponse
from app.schemas.booking import BookingAdminUpdate, BookingCreate, BookingDetail, BookingListItem, BookingResponse, PaymentInfo
from app.schemas.event import AdminOverview, CalendarDay, EventCard, EventCreate, EventDetail, EventUpdate
from app.services.payments import build_payment_payload


router = APIRouter()


def reserved_seats_for_event(db: Session, event_id: int) -> int:
    reserved = db.scalar(
        select(func.coalesce(func.sum(Booking.seats), 0)).where(
            Booking.event_id == event_id,
            Booking.status.in_([BookingStatus.pending_payment.value, BookingStatus.paid.value]),
        )
    )
    return int(reserved or 0)


def serialize_event(db: Session, event: Event) -> EventCard:
    booked_seats = reserved_seats_for_event(db, event.id)
    return EventCard(
        id=event.id,
        slug=event.slug,
        title=event.title,
        short_description=event.short_description,
        venue=event.venue,
        starts_at=event.starts_at,
        ends_at=event.ends_at,
        capacity=event.capacity,
        price_amount=event.price_amount,
        currency=event.currency,
        status=event.status,
        hero_note=event.hero_note,
        booked_seats=booked_seats,
        spots_left=max(event.capacity - booked_seats, 0),
    )


def serialize_payment(payment: Payment | None) -> PaymentInfo | None:
    if payment is None:
        return None
    return PaymentInfo(
        method=payment.method,
        status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        qr_payload=payment.qr_payload,
        recipient_name=payment.recipient_name,
        recipient_bank=payment.recipient_bank,
        recipient_phone=payment.recipient_phone,
        recipient_account=payment.recipient_account,
        recipient_bic=payment.recipient_bic,
        payment_purpose=payment.payment_purpose,
    )


def serialize_booking_detail(db: Session, booking: Booking) -> BookingDetail:
    return BookingDetail(
        id=booking.id,
        booking_token=booking.booking_token,
        customer_name=booking.customer_name,
        phone=booking.phone,
        email=booking.email,
        seats=booking.seats,
        status=booking.status,
        payment_status=booking.payment_status,
        amount_due=booking.amount_due,
        customer_comment=booking.customer_comment,
        created_at=booking.created_at,
        event=serialize_event(db, booking.event),
        payment=serialize_payment(booking.payment),
    )


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    admin = db.scalar(select(AdminUser).where(AdminUser.email == payload.email, AdminUser.is_active.is_(True)))
    if admin is None or not verify_password(payload.password, admin.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    return TokenResponse(access_token=create_access_token(admin.email), admin_name=admin.name)


@router.get("/public/events", response_model=list[EventCard])
def list_public_events(
    db: Session = Depends(get_db),
    include_past: bool = Query(default=False),
) -> list[EventCard]:
    query = select(Event).where(Event.status == EventStatus.published.value)
    if not include_past:
        query = query.where(Event.starts_at >= datetime.now(timezone.utc))
    events = db.scalars(query.order_by(Event.starts_at.asc())).all()
    return [serialize_event(db, event) for event in events]


@router.get("/public/events/calendar", response_model=list[CalendarDay])
def public_calendar(
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    db: Session = Depends(get_db),
) -> list[CalendarDay]:
    start = datetime.strptime(f"{month}-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_month = start.month + 1
    end_year = start.year
    if end_month == 13:
        end_month = 1
        end_year += 1
    end = start.replace(year=end_year, month=end_month)

    events = db.scalars(
        select(Event)
        .where(
            Event.status == EventStatus.published.value,
            Event.starts_at >= start,
            Event.starts_at < end,
        )
        .order_by(Event.starts_at.asc())
    ).all()

    grouped: dict[str, list[EventCard]] = defaultdict(list)
    for event in events:
        grouped[event.starts_at.date().isoformat()].append(serialize_event(db, event))

    return [CalendarDay(date=date_key, events=items) for date_key, items in sorted(grouped.items())]


@router.get("/public/events/{slug}", response_model=EventDetail)
def get_public_event(slug: str, db: Session = Depends(get_db)) -> EventDetail:
    event = db.scalar(select(Event).where(Event.slug == slug))
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    event_card = serialize_event(db, event)
    return EventDetail(**event_card.model_dump(), description=event.description)


@router.post("/public/events/{event_id}/bookings", response_model=BookingResponse)
def create_booking(event_id: int, payload: BookingCreate, db: Session = Depends(get_db)) -> BookingResponse:
    event = db.scalar(select(Event).where(Event.id == event_id).with_for_update())
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    if event.status != EventStatus.published.value:
        raise HTTPException(status_code=400, detail="Booking is available only for published events.")

    booked_seats = reserved_seats_for_event(db, event.id)
    if booked_seats + payload.seats > event.capacity:
        return BookingResponse(
            status=BookingStatus.sold_out.value,
            message="Свободных мест больше нет.",
            event=serialize_event(db, event),
            payment=None,
        )

    amount_due = Decimal(event.price_amount) * payload.seats
    booking = Booking(
        event_id=event.id,
        customer_name=payload.customer_name.strip(),
        phone=payload.phone.strip(),
        email=payload.email.lower(),
        seats=payload.seats,
        amount_due=amount_due,
        customer_comment=payload.customer_comment,
    )
    db.add(booking)
    db.flush()

    payment = Payment(
        booking_id=booking.id,
        amount=amount_due,
        currency=event.currency,
        qr_payload=build_payment_payload(event.title, amount_due, booking.booking_token),
        recipient_name=settings.payment_recipient_name,
        recipient_bank=settings.payment_bank_name,
        recipient_phone=settings.payment_phone,
        recipient_account=settings.payment_account_number or None,
        recipient_bic=settings.payment_bic or None,
        payment_purpose=f"{settings.payment_note_prefix} {event.title} #{booking.booking_token[:8]}",
    )
    db.add(payment)
    db.commit()
    db.refresh(booking)

    return BookingResponse(
        status=BookingStatus.pending_payment.value,
        message="Бронь создана и ожидает оплаты.",
        booking_token=booking.booking_token,
        booking_id=booking.id,
        event=serialize_event(db, event),
        payment=serialize_payment(payment),
    )


@router.get("/public/bookings/{booking_token}", response_model=BookingDetail)
def get_booking(booking_token: str, db: Session = Depends(get_db)) -> BookingDetail:
    booking = db.scalar(
        select(Booking)
        .options(joinedload(Booking.event), joinedload(Booking.payment))
        .where(Booking.booking_token == booking_token)
    )
    if booking is None:
        raise HTTPException(status_code=404, detail="Booking not found.")
    return serialize_booking_detail(db, booking)


@router.get("/admin/overview", response_model=AdminOverview)
def admin_overview(db: Session = Depends(get_db), _: AdminUser = Depends(get_current_admin)) -> AdminOverview:
    total_events = int(db.scalar(select(func.count()).select_from(Event)) or 0)
    published_events = int(
        db.scalar(select(func.count()).select_from(Event).where(Event.status == EventStatus.published.value)) or 0
    )
    pending_bookings = int(
        db.scalar(select(func.count()).select_from(Booking).where(Booking.status == BookingStatus.pending_payment.value)) or 0
    )
    paid_bookings = int(
        db.scalar(select(func.count()).select_from(Booking).where(Booking.status == BookingStatus.paid.value)) or 0
    )
    revenue_paid = db.scalar(
        select(func.coalesce(func.sum(Payment.amount), 0)).where(Payment.status == PaymentStatus.paid.value)
    ) or Decimal("0")
    return AdminOverview(
        total_events=total_events,
        published_events=published_events,
        pending_bookings=pending_bookings,
        paid_bookings=paid_bookings,
        revenue_paid=Decimal(revenue_paid),
    )


@router.get("/admin/events", response_model=list[EventDetail])
def admin_events(db: Session = Depends(get_db), _: AdminUser = Depends(get_current_admin)) -> list[EventDetail]:
    events = db.scalars(select(Event).order_by(Event.starts_at.asc())).all()
    return [EventDetail(**serialize_event(db, event).model_dump(), description=event.description) for event in events]


@router.post("/admin/events", response_model=EventDetail)
def create_event(
    payload: EventCreate,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
) -> EventDetail:
    slug_base = slugify(payload.title)
    slug = slug_base
    suffix = 2
    while db.scalar(select(Event.id).where(Event.slug == slug)):
        slug = f"{slug_base}-{suffix}"
        suffix += 1

    published_at = datetime.now(timezone.utc) if payload.status == EventStatus.published.value else None
    event = Event(
        slug=slug,
        title=payload.title.strip(),
        short_description=payload.short_description.strip(),
        description=payload.description.strip(),
        venue=payload.venue.strip(),
        starts_at=payload.starts_at,
        ends_at=payload.ends_at,
        capacity=payload.capacity,
        price_amount=payload.price_amount,
        status=payload.status,
        hero_note=payload.hero_note,
        published_at=published_at,
        archived_at=datetime.now(timezone.utc) if payload.status == EventStatus.archived.value else None,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    event_card = serialize_event(db, event)
    return EventDetail(**event_card.model_dump(), description=event.description)


@router.patch("/admin/events/{event_id}", response_model=EventDetail)
def update_event(
    event_id: int,
    payload: EventUpdate,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
) -> EventDetail:
    event = db.scalar(select(Event).where(Event.id == event_id))
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(event, key, value)

    if "title" in data:
        event.slug = slugify(event.title)
    if data.get("status") == EventStatus.published.value and event.published_at is None:
        event.published_at = datetime.now(timezone.utc)
    if data.get("status") == EventStatus.archived.value:
        event.archived_at = datetime.now(timezone.utc)

    db.add(event)
    db.commit()
    db.refresh(event)
    event_card = serialize_event(db, event)
    return EventDetail(**event_card.model_dump(), description=event.description)


@router.get("/admin/events/{event_id}/bookings", response_model=list[BookingListItem])
def admin_event_bookings(
    event_id: int,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(get_current_admin),
) -> list[BookingListItem]:
    event = db.scalar(select(Event).where(Event.id == event_id))
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    bookings = db.scalars(
        select(Booking)
        .options(joinedload(Booking.event))
        .where(Booking.event_id == event_id)
        .order_by(Booking.created_at.desc())
    ).all()
    return [
        BookingListItem(
            id=item.id,
            booking_token=item.booking_token,
            customer_name=item.customer_name,
            phone=item.phone,
            email=item.email,
            seats=item.seats,
            status=item.status,
            payment_status=item.payment_status,
            amount_due=item.amount_due,
            created_at=item.created_at,
            event_title=item.event.title,
            event_id=item.event.id,
        )
        for item in bookings
    ]


@router.get("/admin/bookings", response_model=list[BookingListItem])
def admin_bookings(db: Session = Depends(get_db), _: AdminUser = Depends(get_current_admin)) -> list[BookingListItem]:
    bookings = db.scalars(
        select(Booking).options(joinedload(Booking.event)).order_by(Booking.created_at.desc())
    ).all()
    return [
        BookingListItem(
            id=item.id,
            booking_token=item.booking_token,
            customer_name=item.customer_name,
            phone=item.phone,
            email=item.email,
            seats=item.seats,
            status=item.status,
            payment_status=item.payment_status,
            amount_due=item.amount_due,
            created_at=item.created_at,
            event_title=item.event.title,
            event_id=item.event.id,
        )
        for item in bookings
    ]


@router.patch("/admin/bookings/{booking_id}", response_model=BookingDetail)
def update_booking(
    booking_id: int,
    payload: BookingAdminUpdate,
    db: Session = Depends(get_db),
    admin: AdminUser = Depends(get_current_admin),
) -> BookingDetail:
    booking = db.scalar(
        select(Booking)
        .options(joinedload(Booking.event), joinedload(Booking.payment))
        .where(Booking.id == booking_id)
    )
    if booking is None:
        raise HTTPException(status_code=404, detail="Booking not found.")

    if payload.status is not None:
        booking.status = payload.status
    if payload.payment_status is not None:
        booking.payment_status = payload.payment_status
        if booking.payment is not None:
            booking.payment.status = payload.payment_status
            booking.payment.confirmed_by_admin_id = admin.id
            if payload.payment_status == PaymentStatus.paid.value:
                booking.payment.paid_at = datetime.now(timezone.utc)
                booking.status = BookingStatus.paid.value
            if payload.payment_status == PaymentStatus.cancelled.value:
                booking.status = BookingStatus.cancelled.value

    db.add(booking)
    if booking.payment is not None:
        db.add(booking.payment)
    db.commit()
    db.refresh(booking)
    return serialize_booking_detail(db, booking)
