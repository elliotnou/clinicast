from datetime import datetime
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import HIGH_RISK_THRESHOLD, MODEL_PATH
from app.database import Base, engine, get_db
from app.models import Appointment, Patient, Reminder
from app.schemas import (
    HighRiskAppointment,
    PredictRequest,
    PredictResponse,
    ReminderRequest,
    ReminderResponse,
)

# loaded at startup
model_bundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    try:
        model_bundle = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Warning: model not found at {MODEL_PATH}, /predict will fail")
        model_bundle = None
    yield


app = FastAPI(
    title="Clinicast",
    description="Clinic no-show prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_patient_noshow_rate(db: Session, patient_id: int) -> float:
    """Look up the patient's historical no-show rate."""
    result = db.execute(
        text("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'no-show') AS noshows,
                COUNT(*) AS total
            FROM appointments
            WHERE patient_id = :pid
              AND status IN ('showed', 'no-show')
        """),
        {"pid": patient_id},
    ).fetchone()

    if result is None or result.total == 0:
        return 0.20  # prior for new patients
    return result.noshows / result.total


def _encode_appointment_type(appt_type: str) -> int:
    """Simple encoding — matches what the label encoder produces on sorted unique values."""
    if model_bundle and "label_encoder" in model_bundle:
        le = model_bundle["label_encoder"]
        return int(le.transform([appt_type])[0])
    # fallback: alphabetical order (GP=0, telehealth=1, walk-in=2)
    mapping = {"GP": 0, "telehealth": 1, "walk-in": 2}
    return mapping.get(appt_type, 0)


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


def _build_features(req: PredictRequest, db: Session) -> np.ndarray:
    patient = db.query(Patient).filter(Patient.patient_id == req.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {req.patient_id} not found")

    dt = req.scheduled_datetime
    noshow_rate = _get_patient_noshow_rate(db, req.patient_id)

    features = [
        req.booked_lead_time_days,
        dt.weekday(),
        dt.hour,
        _encode_appointment_type(req.appointment_type),
        noshow_rate,
        req.reminders_sent,
        _age_to_group(patient.age),
    ]
    return np.array([features])


@app.post("/predict", response_model=PredictResponse)
def predict_noshow(req: PredictRequest, db: Session = Depends(get_db)):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = _build_features(req, db)
    model = model_bundle["model"]
    prob = float(model.predict_proba(features)[0][1])

    if prob >= HIGH_RISK_THRESHOLD:
        risk = "high"
    elif prob >= 0.3:
        risk = "medium"
    else:
        risk = "low"

    return PredictResponse(
        appointment_type=req.appointment_type,
        noshow_probability=round(prob, 4),
        risk_level=risk,
    )


@app.get("/appointments/high-risk", response_model=list[HighRiskAppointment])
def get_high_risk_appointments(db: Session = Depends(get_db)):
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
    return appointments


@app.post("/reminders/trigger", response_model=ReminderResponse)
def trigger_reminder(req: ReminderRequest, db: Session = Depends(get_db)):
    appt = (
        db.query(Appointment)
        .filter(Appointment.appointment_id == req.appointment_id)
        .first()
    )
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    reminder = Reminder(
        appointment_id=req.appointment_id,
        channel=req.channel,
        sent_at=datetime.now(),
        opened=False,
    )
    db.add(reminder)

    appt.reminders_sent = (appt.reminders_sent or 0) + 1
    db.commit()
    db.refresh(reminder)

    return ReminderResponse(
        reminder_id=reminder.reminder_id,
        appointment_id=reminder.appointment_id,
        channel=reminder.channel,
        sent_at=reminder.sent_at,
        new_reminder_count=appt.reminders_sent,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_bundle is not None}
