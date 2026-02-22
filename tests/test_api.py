import numpy as np
import pytest
from datetime import date, datetime

from rest_framework.test import APIClient

from clinic.models import Appointment, Patient


def _make_test_model_bundle():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    rng = np.random.RandomState(42)
    X = rng.rand(100, 7)
    y = (rng.rand(100) > 0.8).astype(int)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    le = LabelEncoder()
    le.fit(["GP", "telehealth", "walk-in"])

    features = [
        "lead_time_days", "day_of_week", "hour_of_day",
        "appointment_type_encoded", "patient_historical_noshow_rate",
        "reminders_sent", "patient_age_group",
    ]
    return {"model": clf, "label_encoder": le, "features": features}


@pytest.fixture(autouse=True)
def _inject_model(monkeypatch):
    from clinic import views
    bundle = _make_test_model_bundle()
    monkeypatch.setattr(views, "_model_bundle", bundle)


@pytest.mark.django_db
class TestHealthEndpoint:
    def test_health_check(self):
        client = APIClient()
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.django_db
class TestPredictEndpoint:
    def setup_method(self):
        self.client = APIClient()
        self.patient = Patient.objects.create(
            age=35, gender="F",
            postal_code="M5V 2T6", registration_date=date(2022, 1, 1),
        )
        Appointment.objects.create(
            patient=self.patient, clinic_id=1,
            scheduled_datetime=datetime(2025, 6, 15, 10, 0),
            appointment_type="GP", booked_lead_time_days=7,
            reminders_sent=0, status="showed",
        )

    def test_predict_returns_probability(self):
        resp = self.client.post("/api/predict", {
            "patient_id": self.patient.pk,
            "scheduled_datetime": "2025-06-20T09:00:00",
            "appointment_type": "GP",
            "booked_lead_time_days": 7,
            "reminders_sent": 1,
        }, format="json")
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["noshow_probability"] <= 1
        assert data["risk_level"] in ("low", "medium", "high")

    def test_predict_invalid_appointment_type(self):
        resp = self.client.post("/api/predict", {
            "patient_id": self.patient.pk,
            "scheduled_datetime": "2025-06-20T09:00:00",
            "appointment_type": "surgery",
            "booked_lead_time_days": 7,
            "reminders_sent": 0,
        }, format="json")
        assert resp.status_code == 400

    def test_predict_missing_fields(self):
        resp = self.client.post("/api/predict", {
            "patient_id": self.patient.pk,
        }, format="json")
        assert resp.status_code == 400

    def test_predict_patient_not_found(self):
        resp = self.client.post("/api/predict", {
            "patient_id": 9999,
            "scheduled_datetime": "2025-06-20T09:00:00",
            "appointment_type": "GP",
            "booked_lead_time_days": 5,
        }, format="json")
        assert resp.status_code == 404

    def test_predict_negative_lead_time(self):
        resp = self.client.post("/api/predict", {
            "patient_id": self.patient.pk,
            "scheduled_datetime": "2025-06-20T09:00:00",
            "appointment_type": "telehealth",
            "booked_lead_time_days": -1,
        }, format="json")
        assert resp.status_code == 400


@pytest.mark.django_db
class TestReminderEndpoint:
    def setup_method(self):
        self.client = APIClient()
        self.patient = Patient.objects.create(
            age=35, gender="F",
            postal_code="M5V 2T6", registration_date=date(2022, 1, 1),
        )
        self.appt = Appointment.objects.create(
            patient=self.patient, clinic_id=1,
            scheduled_datetime=datetime(2025, 6, 15, 10, 0),
            appointment_type="GP", booked_lead_time_days=7,
            reminders_sent=0, status="showed",
        )

    def test_trigger_reminder(self):
        resp = self.client.post("/api/reminders/trigger", {
            "appointment_id": self.appt.pk,
            "channel": "SMS",
        }, format="json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["appointment_id"] == self.appt.pk
        assert data["channel"] == "SMS"
        assert data["new_reminder_count"] == 1

    def test_trigger_reminder_appointment_not_found(self):
        resp = self.client.post("/api/reminders/trigger", {
            "appointment_id": 9999,
            "channel": "email",
        }, format="json")
        assert resp.status_code == 404


@pytest.mark.django_db
class TestHighRiskEndpoint:
    def setup_method(self):
        self.client = APIClient()
        self.patient = Patient.objects.create(
            age=35, gender="F",
            postal_code="M5V 2T6", registration_date=date(2022, 1, 1),
        )

    def test_high_risk_returns_list(self):
        Appointment.objects.create(
            patient=self.patient, clinic_id=1,
            scheduled_datetime=datetime(2026, 12, 1, 10, 0),
            appointment_type="GP", booked_lead_time_days=30,
            reminders_sent=0, status="showed", noshow_probability=0.85,
        )
        resp = self.client.get("/api/appointments/high-risk")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["noshow_probability"] >= 0.6
