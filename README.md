# Clinicast

No-show prediction system for clinic appointments. Flags high-risk bookings so staff can send targeted reminders instead of guessing.

Built for Canadian primary care settings where no-show rates sit around 18-22%.

## How it works

1. Synthetic dataset of ~10k appointments with realistic patterns (lead time, day of week, patient history, reminders)
2. Random Forest classifier trained on those patterns — predicts no-show probability per appointment
3. Django REST Framework API to score appointments in real time and list high-risk bookings
4. Dashboard showing dataset stats, model performance, and a live scoring form

## Stack

Python 3.11 · Django · Django REST Framework · Strawberry GraphQL · PostgreSQL · scikit-learn · Docker · GitHub Actions

## Quick start

```bash
docker compose up --build -d

docker compose exec app python manage.py generate_data
docker compose exec app python scripts/train_model.py
docker compose exec app python manage.py score_appointments
docker compose restart app
```

Dashboard at `localhost:8000`. Admin panel at `localhost:8000/admin/`.

## API

**`POST /api/predict`** — score an appointment

```json
{
  "patient_id": 42,
  "scheduled_datetime": "2025-06-20T09:00:00",
  "appointment_type": "GP",
  "booked_lead_time_days": 14,
  "reminders_sent": 0
}
```

```json
{
  "appointment_type": "GP",
  "noshow_probability": 0.65,
  "risk_level": "high"
}
```

**`GET /api/appointments/high-risk`** — upcoming appointments scoring >= 0.6

**`POST /api/reminders/trigger`** — log a reminder sent for an appointment

**`GET /api/health`** — health check + model status

GraphQL is available at `/graphql` with the same operations.

## Model

Random Forest (200 trees, max_depth=12, class_weight=balanced). AUC ~0.71 on held-out test set.

Features: booking lead time, day of week, hour, appointment type, patient historical no-show rate, reminders sent, age group. Historical no-show rate uses a look-back approach to avoid data leakage.

Logistic Regression baseline included for comparison (AUC ~0.69).

## Project structure

```
manage.py
clinicast/              # Django project config
clinic/                 # Main app — models, views, serializers, admin, GraphQL
  management/commands/  # generate_data, score_appointments
  templates/            # Dashboard
scripts/                # ML training pipeline
models/                 # Trained model artifacts (gitignored)
tests/                  # API tests (pytest-django, SQLite)
```

## Limitations

- Synthetic data only — patterns are hand-tuned to be plausible but would need real clinic validation
- No weather, provider-specific, or seasonal features
- No auth — meant as a demo, not production
- Reminder triggering is a DB write, no actual SMS/email
