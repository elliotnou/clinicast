# Clinicast

Predicts patient no-shows for clinic appointments using historical booking patterns. Built for Canadian primary care settings where no-show rates typically sit around 18-22%.

The system flags high-risk appointments so staff can send targeted reminders before it's too late. Not a research project — this is meant to be a practical tool that could plug into a real clinic workflow.

## What it does

- Generates ~10,000 synthetic appointments with realistic no-show patterns (lead time, day of week, patient history, reminders sent)
- Trains a Random Forest classifier to predict no-show probability per appointment
- Exposes a FastAPI backend with endpoints to score appointments, list high-risk bookings, and trigger reminders
- Runs in Docker — one command to spin up Postgres + the API

## Tech stack

Python 3.11, FastAPI, PostgreSQL, SQLAlchemy, scikit-learn, pandas, Docker, GitHub Actions

## Quick start

### With Docker (recommended)

```bash
# start postgres and the API
docker compose up --build -d

# generate data and train model
docker compose exec app python -m scripts.generate_data
docker compose exec app python -m scripts.train_model

# restart app so it loads the trained model
docker compose restart app
```

API is at http://localhost:8000. Docs at http://localhost:8000/docs.

### Local development

```bash
# install deps
pip install -r requirements.txt

# start postgres (or use docker just for the db)
docker compose up db -d

# copy and edit env
cp .env.example .env

# generate data + train
python -m scripts.generate_data
python -m scripts.train_model

# run the API
uvicorn app.main:app --reload
```

## API endpoints

### `POST /predict`

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

### `GET /appointments/high-risk`

Returns all upcoming appointments with noshow_probability >= 0.6, sorted by time.

### `POST /reminders/trigger`

Log a reminder sent for an appointment. Updates the reminder count on the appointment record.

```json
{
  "appointment_id": 1,
  "channel": "SMS"
}
```

## Model details

Two models are trained for comparison:

- **Logistic Regression** — baseline
- **Random Forest** (200 trees, max_depth=12) — used in production

Features: lead time, day of week, hour, appointment type, patient historical no-show rate, reminders sent, age group.

The historical no-show rate is computed with a look-back approach (no data leakage — each appointment only sees past visits for that patient).

Target AUC: 0.75-0.80 on held-out test set.

## Data generation

The synthetic dataset bakes in patterns that match what you'd see in real clinic data:

- Longer lead times → more no-shows
- Monday mornings, Friday afternoons → more no-shows
- Reminders sent → fewer no-shows
- New patients → more no-shows than established ones
- Walk-ins → almost never no-show (they're already there)

See `scripts/generate_data.py` for the full logic.

## Tests

```bash
pytest tests/ -v
```

Tests use an in-memory SQLite database — no Postgres needed. Covers prediction endpoint, validation, error cases, and the reminder flow.

## Project structure

```
clinicast/
├── app/
│   ├── config.py          # env vars and settings
│   ├── database.py        # sqlalchemy engine/session
│   ├── main.py            # fastapi app + endpoints
│   ├── models.py          # orm models (patients, appointments, reminders)
│   └── schemas.py         # pydantic request/response schemas
├── scripts/
│   ├── generate_data.py   # synthetic data generation
│   ├── train_model.py     # ml training pipeline
│   └── setup.sh           # convenience script
├── tests/
│   ├── conftest.py        # fixtures + test db setup
│   └── test_api.py        # api tests
├── models/                # trained model artifacts (gitignored)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Limitations

- Uses synthetic data, not real patient records. The patterns are hand-tuned to be plausible, but a real deployment would need validation against actual clinic data.
- The model doesn't account for weather, provider-specific patterns, or seasonal trends — all of which matter in practice.
- No auth on the API. In production you'd want this behind clinic network auth at minimum.
- Reminder triggering is just a database write — no actual SMS/email integration.
