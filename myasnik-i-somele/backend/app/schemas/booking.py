from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.schemas.event import EventCard


class BookingCreate(BaseModel):
    customer_name: str = Field(min_length=2, max_length=120)
    phone: str = Field(min_length=6, max_length=40)
    email: EmailStr
    seats: int = Field(default=1, ge=1, le=8)
    customer_comment: str | None = Field(default=None, max_length=500)


class PaymentInfo(BaseModel):
    method: str
    status: str
    amount: Decimal
    currency: str
    qr_payload: str
    recipient_name: str
    recipient_bank: str
    recipient_phone: str
    recipient_account: str | None
    recipient_bic: str | None
    payment_purpose: str


class BookingResponse(BaseModel):
    status: str
    message: str
    booking_token: str | None = None
    booking_id: int | None = None
    event: EventCard
    payment: PaymentInfo | None = None


class BookingDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    booking_token: str
    customer_name: str
    phone: str
    email: str
    seats: int
    status: str
    payment_status: str
    amount_due: Decimal
    customer_comment: str | None
    created_at: datetime
    event: EventCard
    payment: PaymentInfo | None = None


class BookingAdminUpdate(BaseModel):
    status: str | None = None
    payment_status: str | None = None


class BookingListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    booking_token: str
    customer_name: str
    phone: str
    email: str
    seats: int
    status: str
    payment_status: str
    amount_due: Decimal
    created_at: datetime
    event_title: str
    event_id: int
