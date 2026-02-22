from datetime import datetime

import strawberry
from django.conf import settings
from strawberry.django.views import GraphQLView

from .models import Appointment, Patient, Reminder


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
        appointments = (
            Appointment.objects
            .filter(
                noshow_probability__gte=settings.HIGH_RISK_THRESHOLD,
                scheduled_datetime__gte=datetime.now(),
            )
            .exclude(status="cancelled")
            .order_by("scheduled_datetime")
        )
        return [
            HighRiskAppointmentType(
                appointment_id=a.pk,
                patient_id=a.patient_id,
                scheduled_datetime=a.scheduled_datetime,
                appointment_type=a.appointment_type,
                noshow_probability=a.noshow_probability,
                reminders_sent=a.reminders_sent,
            )
            for a in appointments
        ]

    @strawberry.field
    def health(self) -> HealthResult:
        from .views import _get_model_bundle

        return HealthResult(
            status="ok",
            model_loaded=_get_model_bundle() is not None,
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    def predict_noshow(self, input: PredictInput) -> PredictResult:
        import numpy as np
        from .views import (
            _get_model_bundle,
            _get_patient_noshow_rate,
            _encode_appointment_type,
            _age_to_group,
        )

        bundle = _get_model_bundle()
        if bundle is None:
            raise ValueError("Model not loaded")

        try:
            patient = Patient.objects.get(pk=input.patient_id)
        except Patient.DoesNotExist:
            raise ValueError(f"Patient {input.patient_id} not found")

        dt = input.scheduled_datetime
        noshow_rate = _get_patient_noshow_rate(input.patient_id)

        features = np.array([[
            input.booked_lead_time_days,
            dt.weekday(),
            dt.hour,
            _encode_appointment_type(input.appointment_type),
            noshow_rate,
            input.reminders_sent,
            _age_to_group(patient.age),
        ]])

        model = bundle["model"]
        prob = float(model.predict_proba(features)[0][1])

        if prob >= settings.HIGH_RISK_THRESHOLD:
            risk = "high"
        elif prob >= 0.3:
            risk = "medium"
        else:
            risk = "low"

        return PredictResult(
            appointment_type=input.appointment_type,
            noshow_probability=round(prob, 4),
            risk_level=risk,
        )

    @strawberry.mutation
    def trigger_reminder(self, input: ReminderInput) -> ReminderResult:
        try:
            appt = Appointment.objects.get(pk=input.appointment_id)
        except Appointment.DoesNotExist:
            raise ValueError("Appointment not found")

        reminder = Reminder.objects.create(
            appointment=appt,
            channel=input.channel,
            sent_at=datetime.now(),
            opened=False,
        )

        appt.reminders_sent = (appt.reminders_sent or 0) + 1
        appt.save(update_fields=["reminders_sent"])

        return ReminderResult(
            reminder_id=reminder.pk,
            appointment_id=reminder.appointment_id,
            channel=reminder.channel,
            sent_at=reminder.sent_at,
            new_reminder_count=appt.reminders_sent,
        )


schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_view = GraphQLView.as_view(schema=schema)
