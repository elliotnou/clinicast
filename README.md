# Clinicast

Predicts patient no-shows for clinic appointments using historical booking patterns. Built for Canadian primary care settings where no-show rates typically sit around 18-22%.

The system flags high-risk appointments so staff can send targeted reminders before it's too late. Not a research project — this is meant to be a practical tool that could plug into a real clinic workflow.

## What it does

- Generates ~10,000 synthetic appointments with realistic no-show patterns (lead time, day of week, patient history, reminders sent)
- Trains a Random Forest classifier to predict no-show probability per appointment
- Serves a Django REST Framework backend with endpoints to score appointments, list high-risk bookings, and trigger reminders
- Includes a dashboard for stats and live scoring
- Runs in Docker — one command to spin up Postgres + the app

## Tech stack

Python 3.11, Django, Django REST Framework, Strawberry GraphQL, PostgreSQL, scikit-learn, pandas, Docker, GitHub Actions

## Quick start

### With Docker (recommended)

```bash
# start postgres and the app
docker compose up --build -d

# generate data, train model, and score appointments
docker compose exec app python manage.py generate_data
docker compose exec app python scripts/train_model.py
docker compose exec app python manage.py score_appointments

# restart app so it loads the trained model
docker compose restart app
```

App is at http://localhost:8000. Admin at http://localhost:8000/admin/.

### Local development

```bash
# install deps
pip install -r requirements.txt

# start postgres (or use docker just for the db)
docker compose up db -d

# copy and edit env
cp .env.example .env

# run migrations
python manage.py migrate

# generate data, train, and score
python manage.py generate_data
python scripts/train_model.py
python manage.py score_appointments

# run the dev server
python manage.py runserver
```

## API endpoints

### `POST /api/predict`

Score an appointment for no-show risk.

```json
{
  "patient_id": 42,
  "scheduled_datetime": "2025-06-20T09:00:00",
  "appointment_type": "GP",
  "booked_lead_time_days": 14,
  "reminders_sent": 0
}
```

Returns:

```json
{
  "appointment_type": "GP",
  "noshow_probability": 0.34,
  "risk_level": "medium"
}
```

### `GET /api/appointments/high-risk`

Returns all upcoming appointments with noshow_probability >= 0.6, sorted by time. Scores are written to the DB by the batch scoring step (`python manage.py score_appointments`), which you'd run on a schedule in production (e.g. daily cron).

### `POST /api/reminders/trigger`

Log a reminder sent for an appointment. Updates the reminder count on the appointment record.

```json
{
  "appointment_id": 1,
  "channel": "SMS"
}
```

## GraphQL API

The same functionality is also exposed via GraphQL at `/graphql`, built with [Strawberry](https://strawberry.rocks/). Hit `/graphql` in the browser to open the GraphiQL playground.

Example query — fetch high-risk appointments:

```graphql
query {
  highRiskAppointments {
    appointmentId
    patientId
    scheduledDatetime
    appointmentType
    noshowProbability
  }
}
```

Example mutation — score an appointment:

```graphql
mutation {
  predictNoshow(input: {
    patientId: 42
    scheduledDatetime: "2025-06-20T09:00:00"
    appointmentType: "GP"
    bookedLeadTimeDays: 14
    remindersSent: 0
  }) {
    noshowProbability
    riskLevel
  }
}
```

## Django Admin

Visit `/admin/` to manage patients, appointments, and reminders through Django's built-in admin interface. Models are registered with searchable, filterable list views.

## Model details

Two models are trained for comparison:

- **Logistic Regression** — baseline
- **Random Forest** (200 trees, max_depth=12, class_weight=balanced) — used in production

Features: lead time, day of week, hour, appointment type, patient historical no-show rate, reminders sent, age group.

The historical no-show rate is computed with a look-back approach (no data leakage — each appointment only sees past visits for that patient). Both models use `class_weight="balanced"` to handle the ~82/18 class split.

AUC on held-out test set: ~0.74 (RF), ~0.70 (LR baseline).

## Data generation

The synthetic dataset bakes in patterns that match what you'd see in real clinic data:

- Longer lead times → more no-shows
- Monday mornings, Friday afternoons → more no-shows
- Reminders sent → fewer no-shows
- New patients → more no-shows than established ones
- Walk-ins → almost never no-show (they're already there)

See `clinic/management/commands/generate_data.py` for the full logic.

## Tests

```bash
pytest tests/ -v
```

Tests use an in-memory SQLite database — no Postgres needed. Covers prediction endpoint, validation, error cases, and the reminder flow.

## Project structure

```
clinicast/
├── manage.py
├── clinicast/                    # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── clinic/                       # Django app
│   ├── models.py                 # Patient, Appointment, Reminder (Django ORM)
│   ├── serializers.py            # DRF serializers
│   ├── views.py                  # DRF views + dashboard
│   ├── urls.py                   # API routing
│   ├── admin.py                  # Django admin config
│   ├── graphql_schema.py         # Strawberry GraphQL schema
│   ├── management/
│   │   └── commands/
│   │       ├── generate_data.py  # seed synthetic data
│   │       └── score_appointments.py  # batch-score appointments
│   └── templates/
│       └── dashboard.html        # stats + live scoring UI
├── scripts/
│   └── train_model.py            # ML training pipeline
├── models/                       # trained model artifacts (gitignored)
├── tests/
│   ├── conftest.py               # pytest-django config
│   └── test_api.py               # API tests
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Limitations

- Uses synthetic data, not real patient records. The patterns are hand-tuned to be plausible, but a real deployment would need validation against actual clinic data.
- The model doesn't account for weather, provider-specific patterns, or seasonal trends — all of which matter in practice.
- No auth on the API. In production you'd want this behind clinic network auth at minimum.
- Reminder triggering is just a database write — no actual SMS/email integration.
