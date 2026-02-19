"""Tests for the FastAPI endpoints."""

from datetime import datetime

from app.models import Appointment, Patient


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_predict_returns_probability(client):
    resp = client.post("/predict", json={
        "patient_id": 1,
        "scheduled_datetime": "2025-06-20T09:00:00",
        "appointment_type": "GP",
        "booked_lead_time_days": 7,
        "reminders_sent": 1,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert 0 <= data["noshow_probability"] <= 1
    assert data["risk_level"] in ("low", "medium", "high")


def test_predict_invalid_appointment_type(client):
    resp = client.post("/predict", json={
        "patient_id": 1,
        "scheduled_datetime": "2025-06-20T09:00:00",
        "appointment_type": "surgery",  # not a valid type
        "booked_lead_time_days": 7,
        "reminders_sent": 0,
    })
    assert resp.status_code == 422


def test_predict_missing_fields(client):
    resp = client.post("/predict", json={
        "patient_id": 1,
        # missing scheduled_datetime, appointment_type, lead_time
    })
    assert resp.status_code == 422


def test_predict_patient_not_found(client):
    resp = client.post("/predict", json={
        "patient_id": 9999,
        "scheduled_datetime": "2025-06-20T09:00:00",
        "appointment_type": "GP",
        "booked_lead_time_days": 5,
    })
    assert resp.status_code == 404


def test_predict_negative_lead_time(client):
    resp = client.post("/predict", json={
        "patient_id": 1,
        "scheduled_datetime": "2025-06-20T09:00:00",
        "appointment_type": "telehealth",
        "booked_lead_time_days": -1,
    })
    assert resp.status_code == 422


def test_trigger_reminder(client):
    resp = client.post("/reminders/trigger", json={
        "appointment_id": 1,
        "channel": "SMS",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["appointment_id"] == 1
    assert data["channel"] == "SMS"
    assert data["new_reminder_count"] == 1


def test_trigger_reminder_appointment_not_found(client):
    resp = client.post("/reminders/trigger", json={
        "appointment_id": 9999,
        "channel": "email",
    })
    assert resp.status_code == 404


def test_high_risk_returns_list(client, db):
    # insert an appointment with high noshow_probability in the future
    future_appt = Appointment(
        appointment_id=100, patient_id=1, clinic_id=1,
        scheduled_datetime=datetime(2026, 12, 1, 10, 0),
        appointment_type="GP", booked_lead_time_days=30,
        reminders_sent=0, status="showed", noshow_probability=0.85,
    )
    db.add(future_appt)
    db.commit()

    resp = client.get("/appointments/high-risk")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["noshow_probability"] >= 0.6
