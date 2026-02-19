import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://clinicast:clinicast@localhost:5432/clinicast",
)
MODEL_PATH = os.getenv("MODEL_PATH", "models/noshow_model.joblib")
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.6"))
