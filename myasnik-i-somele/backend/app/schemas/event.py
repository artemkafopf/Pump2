from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class EventBase(BaseModel):
    title: str = Field(min_length=3, max_length=255)
    short_description: str = Field(min_length=8, max_length=280)
    description: str = Field(min_length=16)
    venue: str = Field(min_length=3, max_length=255)
    starts_at: datetime
    ends_at: datetime | None = None
    capacity: int = Field(ge=1, le=500)
    price_amount: Decimal = Field(ge=0, decimal_places=2, max_digits=10)
    status: str = Field(default="draft")
    hero_note: str | None = Field(default=None, max_length=140)


class EventCreate(EventBase):
    pass


class EventUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=3, max_length=255)
    short_description: str | None = Field(default=None, min_length=8, max_length=280)
    description: str | None = Field(default=None, min_length=16)
    venue: str | None = Field(default=None, min_length=3, max_length=255)
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    capacity: int | None = Field(default=None, ge=1, le=500)
    price_amount: Decimal | None = Field(default=None, ge=0, decimal_places=2, max_digits=10)
    status: str | None = None
    hero_note: str | None = Field(default=None, max_length=140)


class EventCard(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    slug: str
    title: str
    short_description: str
    venue: str
    starts_at: datetime
    ends_at: datetime | None
    capacity: int
    price_amount: Decimal
    currency: str
    status: str
    hero_note: str | None
    booked_seats: int
    spots_left: int


class EventDetail(EventCard):
    description: str


class CalendarDay(BaseModel):
    date: str
    events: list[EventCard]


class AdminOverview(BaseModel):
    total_events: int
    published_events: int
    pending_bookings: int
    paid_bookings: int
    revenue_paid: Decimal
