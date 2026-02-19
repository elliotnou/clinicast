"""
Batch-score upcoming appointments and save no-show probabilities to the DB.

Meant to run as a daily job (cron, scheduled task, etc.) so the
/appointments/high-risk endpoint has fresh scores to surface.
"""

import joblib
import numpy as np
from sqlalchemy import text

from app.config import MODEL_PATH
from app.database import SessionLocal
from app.models import Appointment, Patient


def _age_to_group(age: int) -> int:
    if age <= 17:
        return 0
    elif age <= 30:
        return 1
    elif age <= 50:
        return 2
    elif age <= 65:
        return 3
    return 4


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    le = bundle["label_encoder"]

    session = SessionLocal()
    try:
        # grab all non-cancelled appointments that haven't been scored yet
        # or that we want to re-score
        appointments = (
            session.query(Appointment, Patient)
            .join(Patient, Appointment.patient_id == Patient.patient_id)
            .filter(Appointment.status.in_(["showed", "no-show"]))
            .all()
        )

        scored = 0
        for appt, patient in appointments:
            # look up historical no-show rate for this patient
            result = session.execute(
                text("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'no-show') AS noshows,
                        COUNT(*) AS total
                    FROM appointments
                    WHERE patient_id = :pid
                      AND status IN ('showed', 'no-show')
                      AND scheduled_datetime < :cutoff
                """),
                {"pid": patient.patient_id, "cutoff": appt.scheduled_datetime},
            ).fetchone()

            if result.total == 0:
                hist_rate = 0.20
            else:
                hist_rate = result.noshows / result.total

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
            scored += 1

        session.commit()
        print(f"Scored {scored} appointments")

        # quick summary
        high_risk = sum(
            1 for appt, _ in appointments
            if appt.noshow_probability and appt.noshow_probability >= 0.6
        )
        print(f"  {high_risk} flagged as high-risk (>= 0.6)")

    finally:
        session.close()


if __name__ == "__main__":
    main()
