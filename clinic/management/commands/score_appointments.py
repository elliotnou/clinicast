"""
Batch-score appointments and save no-show probabilities to the DB.

Meant to run as a daily job so the high-risk endpoint has fresh scores.
"""

import joblib
import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from clinic.models import Appointment


def _age_to_group(age):
    if age <= 17:
        return 0
    elif age <= 30:
        return 1
    elif age <= 50:
        return 2
    elif age <= 65:
        return 3
    return 4


class Command(BaseCommand):
    help = "Score all appointments with the trained model"

    def handle(self, *args, **options):
        bundle = joblib.load(settings.MODEL_PATH)
        model = bundle["model"]
        le = bundle["label_encoder"]

        appointments = (
            Appointment.objects
            .filter(status__in=["showed", "no-show"])
            .select_related("patient")
        )

        scored = 0
        for appt in appointments:
            patient = appt.patient

            # historical no-show rate (look-back only)
            past = Appointment.objects.filter(
                patient=patient,
                status__in=["showed", "no-show"],
                scheduled_datetime__lt=appt.scheduled_datetime,
            )
            total = past.count()
            if total == 0:
                hist_rate = 0.20
            else:
                hist_rate = past.filter(status="no-show").count() / total

            features = np.array([[
                appt.booked_lead_time_days,
                appt.scheduled_datetime.weekday(),
                appt.scheduled_datetime.hour,
                int(le.transform([appt.appointment_type])[0]),
                hist_rate,
                appt.reminders_sent,
                _age_to_group(patient.age),
            ]])

            prob = float(model.predict_proba(features)[0][1])
            appt.noshow_probability = round(prob, 4)
            appt.save(update_fields=["noshow_probability"])
            scored += 1

        self.stdout.write(f"Scored {scored} appointments")

        high_risk = Appointment.objects.filter(
            noshow_probability__gte=settings.HIGH_RISK_THRESHOLD
        ).count()
        self.stdout.write(f"  {high_risk} flagged as high-risk (>= {settings.HIGH_RISK_THRESHOLD})")
