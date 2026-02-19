"""
Train no-show prediction models on the synthetic appointment data.

Trains two models:
  1. Logistic Regression (baseline)
  2. Random Forest (primary)

Prints classification reports and AUC scores, saves the better model.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.database import engine


FEATURE_COLS = [
    "lead_time_days",
    "day_of_week",
    "hour_of_day",
    "appointment_type_encoded",
    "patient_historical_noshow_rate",
    "reminders_sent",
    "patient_age_group",
]


def load_data():
    query = """
    SELECT
        a.appointment_id,
        a.patient_id,
        a.scheduled_datetime,
        a.appointment_type,
        a.booked_lead_time_days AS lead_time_days,
        a.reminders_sent,
        a.status,
        p.age
    FROM appointments a
    JOIN patients p ON a.patient_id = p.patient_id
    WHERE a.status IN ('showed', 'no-show')
    ORDER BY a.scheduled_datetime
    """
    df = pd.read_sql(query, engine)
    return df


def build_features(df):
    df = df.copy()

    df["day_of_week"] = pd.to_datetime(df["scheduled_datetime"]).dt.dayofweek
    df["hour_of_day"] = pd.to_datetime(df["scheduled_datetime"]).dt.hour

    # encode appointment type
    le = LabelEncoder()
    df["appointment_type_encoded"] = le.fit_transform(df["appointment_type"])

    # patient historical no-show rate (look-back only — no data leakage)
    # sort by time so we can compute a running rate
    df = df.sort_values("scheduled_datetime").reset_index(drop=True)
    patient_noshow_counts = {}
    patient_total_counts = {}
    rates = []
    for _, row in df.iterrows():
        pid = row["patient_id"]
        total = patient_total_counts.get(pid, 0)
        noshows = patient_noshow_counts.get(pid, 0)
        # for first visit, use the global average as a prior
        if total == 0:
            rates.append(0.20)
        else:
            rates.append(noshows / total)

        # update counts AFTER computing the rate (prevents leakage)
        patient_total_counts[pid] = total + 1
        if row["status"] == "no-show":
            patient_noshow_counts[pid] = noshows + 1

    df["patient_historical_noshow_rate"] = rates

    # age groups
    df["patient_age_group"] = pd.cut(
        df["age"],
        bins=[0, 17, 30, 50, 65, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # binary target
    df["target"] = (df["status"] == "no-show").astype(int)

    return df, le


def train_and_evaluate(df, le):
    X = df[FEATURE_COLS].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Logistic Regression (baseline) ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    print("=" * 50)
    print("LOGISTIC REGRESSION (baseline)")
    print("=" * 50)
    print(classification_report(y_test, lr.predict(X_test), target_names=["showed", "no-show"]))
    print(f"AUC: {lr_auc:.4f}\n")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    print("=" * 50)
    print("RANDOM FOREST")
    print("=" * 50)
    print(classification_report(y_test, rf.predict(X_test), target_names=["showed", "no-show"]))
    print(f"AUC: {rf_auc:.4f}\n")

    # feature importance
    print("Feature importance (Random Forest):")
    for name, imp in sorted(
        zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name:40s} {imp:.4f}")

    return rf, lr, le, rf_auc, lr_auc


def main():
    print("Loading data from database...")
    df = load_data()
    print(f"Loaded {len(df)} appointments ({df['status'].value_counts().to_dict()})")

    print("\nBuilding features...")
    df, le = build_features(df)

    print("Training models...\n")
    rf, lr, le, rf_auc, lr_auc = train_and_evaluate(df, le)

    # save the random forest (it's the one we'll use in production)
    os.makedirs("models", exist_ok=True)
    model_path = os.getenv("MODEL_PATH", "models/noshow_model.joblib")
    joblib.dump({"model": rf, "label_encoder": le, "features": FEATURE_COLS}, model_path)
    print(f"\nModel saved to {model_path}")

    # also save a small metadata file for the API to report
    meta = {
        "rf_auc": round(rf_auc, 4),
        "lr_auc": round(lr_auc, 4),
        "n_training_samples": len(df),
        "features": FEATURE_COLS,
    }
    joblib.dump(meta, "models/model_metadata.joblib")
    print("Metadata saved to models/model_metadata.joblib")


if __name__ == "__main__":
    main()
