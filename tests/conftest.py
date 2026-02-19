"""
Test fixtures. Uses an in-memory SQLite database so tests
don't need PostgreSQL running.
"""

import os
import pytest
import numpy as np
from datetime import datetime, date

from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app.models import Patient, Appointment

# use sqlite for tests — no docker needed
TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_model(tmp_path_factory):
    """Create a dummy model so the app can load it."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import joblib

    model_dir = tmp_path_factory.mktemp("models")
    model_path = str(model_dir / "noshow_model.joblib")

    # train a tiny dummy model with the right shape
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
    joblib.dump({"model": clf, "label_encoder": le, "features": features}, model_path)

    os.environ["MODEL_PATH"] = model_path
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    yield


@pytest.fixture()
def db():
    Base.metadata.create_all(bind=test_engine)
    session = TestSession()

    # seed a test patient and appointment
    patient = Patient(
        patient_id=1, age=35, gender="F",
        postal_code="M5V 2T6", registration_date=date(2022, 1, 1),
    )
    appt = Appointment(
        appointment_id=1, patient_id=1, clinic_id=1,
        scheduled_datetime=datetime(2025, 6, 15, 10, 0),
        appointment_type="GP", booked_lead_time_days=7,
        reminders_sent=0, status="showed",
    )
    session.add_all([patient, appt])
    session.commit()

    yield session

    session.close()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def client(db):
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
