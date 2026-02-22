from datetime import datetime
from typing import Optional

import strawberry
from strawberry.fastapi import GraphQLRouter

from app.database import SessionLocal
from app.models import Appointment, Reminder


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@strawberry.type
class HighRiskAppointmentType:
    appointment_id: int
    patient_id: int
    scheduled_datetime: datetime
    appointment_type: str
    noshow_probability: float
    reminders_sent: int


@strawberry.type
class PredictResult:
    appointment_type: str
    noshow_probability: float
    risk_level: str


@strawberry.type
class ReminderResult:
    reminder_id: int
    appointment_id: int
    channel: str
    sent_at: datetime
    new_reminder_count: int


@strawberry.type
class HealthResult:
    status: str
    model_loaded: bool


@strawberry.input
class PredictInput:
    patient_id: int
    scheduled_datetime: datetime
    appointment_type: str
    booked_lead_time_days: int
    reminders_sent: int = 0


@strawberry.input
class ReminderInput:
    appointment_id: int
    channel: str = "SMS"


@strawberry.type
class Query:
    @strawberry.field
    def high_risk_appointments(self) -> list[HighRiskAppointmentType]:
        from app.config import HIGH_RISK_THRESHOLD

        db = SessionLocal()
        try:
            appointments = (
                db.query(Appointment)
                .filter(
                    Appointment.noshow_probability >= HIGH_RISK_THRESHOLD,
                    Appointment.scheduled_datetime >= datetime.now(),
                    Appointment.status != "cancelled",
                )
                .order_by(Appointment.scheduled_datetime)
                .all()
            )
            return [
                HighRiskAppointmentType(
                    appointment_id=a.appointment_id,
                    patient_id=a.patient_id,
                    scheduled_datetime=a.scheduled_datetime,
                    appointment_type=a.appointment_type,
                    noshow_probability=a.noshow_probability,
                    reminders_sent=a.reminders_sent,
                )
                for a in appointments
            ]
        finally:
            db.close()

    @strawberry.field
    def health(self) -> HealthResult:
        from app.main import model_bundle

        return HealthResult(
            status="ok",
            model_loaded=model_bundle is not None,
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    def predict_noshow(self, input: PredictInput) -> PredictResult:
        from app.config import HIGH_RISK_THRESHOLD
        from app.main import model_bundle, _build_features
        from app.schemas import PredictRequest

        if model_bundle is None:
            raise ValueError("Model not loaded")

        req = PredictRequest(
            patient_id=input.patient_id,
            scheduled_datetime=input.scheduled_datetime,
            appointment_type=input.appointment_type,
            booked_lead_time_days=input.booked_lead_time_days,
            reminders_sent=input.reminders_sent,
        )

        db = SessionLocal()
        try:
            import numpy as np

            features = _build_features(req, db)
            model = model_bundle["model"]
            prob = float(model.predict_proba(features)[0][1])

            if prob >= HIGH_RISK_THRESHOLD:
                risk = "high"
            elif prob >= 0.3:
                risk = "medium"
            else:
                risk = "low"

            return PredictResult(
                appointment_type=req.appointment_type,
                noshow_probability=round(prob, 4),
                risk_level=risk,
            )
        finally:
            db.close()

    @strawberry.mutation
    def trigger_reminder(self, input: ReminderInput) -> ReminderResult:
        db = SessionLocal()
        try:
            appt = (
                db.query(Appointment)
                .filter(Appointment.appointment_id == input.appointment_id)
                .first()
            )
            if not appt:
                raise ValueError("Appointment not found")

            reminder = Reminder(
                appointment_id=input.appointment_id,
                channel=input.channel,
                sent_at=datetime.now(),
                opened=False,
            )
            db.add(reminder)
            appt.reminders_sent = (appt.reminders_sent or 0) + 1
            db.commit()
            db.refresh(reminder)

            return ReminderResult(
                reminder_id=reminder.reminder_id,
                appointment_id=reminder.appointment_id,
                channel=reminder.channel,
                sent_at=reminder.sent_at,
                new_reminder_count=appt.reminders_sent,
            )
        finally:
            db.close()


schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)
