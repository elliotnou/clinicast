from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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

templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
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


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    # overall stats
    total = db.execute(text("SELECT COUNT(*) FROM appointments")).scalar()
    noshows = db.execute(
        text("SELECT COUNT(*) FROM appointments WHERE status = 'no-show'")
    ).scalar()
    noshow_rate = (noshows / total * 100) if total > 0 else 0
    high_risk_count = db.execute(
        text("SELECT COUNT(*) FROM appointments WHERE noshow_probability >= :t"),
        {"t": HIGH_RISK_THRESHOLD},
    ).scalar()

    # load model metadata if available
    rf_auc = 0.0
    try:
        meta = joblib.load("models/model_metadata.joblib")
        rf_auc = meta.get("rf_auc", 0.0)
    except Exception:
        pass

    # no-show rate by appointment type
    by_type_rows = db.execute(text("""
        SELECT appointment_type,
               COUNT(*) FILTER (WHERE status = 'no-show') * 100.0 / COUNT(*) AS rate
        FROM appointments
        WHERE status IN ('showed', 'no-show')
        GROUP BY appointment_type
        ORDER BY rate DESC
    """)).fetchall()
    by_type = [{"type": r[0], "rate": float(r[1])} for r in by_type_rows]

    # feature importance from the loaded model
    feature_importance = []
    max_importance = 0.01
    if model_bundle:
        model = model_bundle["model"]
        feature_names = model_bundle.get("features", [])
        display_names = {
            "lead_time_days": "Lead Time",
            "day_of_week": "Day of Week",
            "hour_of_day": "Hour",
            "appointment_type_encoded": "Appt Type",
            "patient_historical_noshow_rate": "Patient History",
            "reminders_sent": "Reminders",
            "patient_age_group": "Age Group",
        }
        importances = model.feature_importances_
        max_importance = max(importances)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            feature_importance.append({
                "name": display_names.get(name, name),
                "importance": float(imp),
            })

    # high-risk appointments (show top 25)
    high_risk = (
        db.query(Appointment)
        .filter(Appointment.noshow_probability >= HIGH_RISK_THRESHOLD)
        .order_by(Appointment.noshow_probability.desc())
        .limit(25)
        .all()
    )

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": {
            "total_appointments": total,
            "noshow_rate": noshow_rate,
            "high_risk_count": high_risk_count,
            "rf_auc": rf_auc,
        },
        "by_type": by_type,
        "feature_importance": feature_importance,
        "max_importance": max_importance,
        "high_risk": high_risk,
    })


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_bundle is not None}
