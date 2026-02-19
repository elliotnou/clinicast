from datetime import datetime
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    patient_id: int
    scheduled_datetime: datetime
    appointment_type: str = Field(
        ..., pattern="^(GP|telehealth|walk-in)$"
    )
    booked_lead_time_days: int = Field(..., ge=0)
    reminders_sent: int = Field(0, ge=0)


class PredictResponse(BaseModel):
    appointment_type: str
    noshow_probability: float
    risk_level: str  # "low", "medium", "high"


class HighRiskAppointment(BaseModel):
    appointment_id: int
    patient_id: int
    scheduled_datetime: datetime
    appointment_type: str
    noshow_probability: float
    reminders_sent: int

    model_config = {"from_attributes": True}


class ReminderRequest(BaseModel):
    appointment_id: int
    channel: str = Field("SMS", pattern="^(SMS|email)$")


class ReminderResponse(BaseModel):
    reminder_id: int
    appointment_id: int
    channel: str
    sent_at: datetime
    new_reminder_count: int
