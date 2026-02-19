"""
Generate synthetic clinic appointment data and seed the database.

Creates ~10k appointments over 2 years with realistic no-show patterns
modelled after Canadian primary care clinics (~18-22% no-show rate).
"""

import random
from datetime import datetime, timedelta

import numpy as np
from faker import Faker
from sqlalchemy import text

from app.database import engine, SessionLocal, Base
from app.models import Patient, Appointment, Reminder

fake = Faker("en_CA")
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# -- tunables --
NUM_PATIENTS = 1200
NUM_APPOINTMENTS = 10000
DATE_START = datetime(2022, 1, 1)
DATE_END = datetime(2023, 12, 31)
CLINIC_IDS = [1, 2, 3]
APPOINTMENT_TYPES = ["GP", "telehealth", "walk-in"]
TYPE_WEIGHTS = [0.55, 0.25, 0.20]

# Canadian postal code prefixes by province (keeps it looking real)
POSTAL_PREFIXES = [
    "M5V", "M4Y", "L5B", "K1A", "N2L", "T2P", "V6B", "R3C", "E1C", "A1B",
]


def generate_patients(session):
    patients = []
    for i in range(1, NUM_PATIENTS + 1):
        reg_date = fake.date_between(
            start_date=DATE_START - timedelta(days=365 * 3),
            end_date=DATE_END,
        )
        p = Patient(
            patient_id=i,
            age=int(np.random.choice(
                range(2, 90),
                p=_age_distribution(),
            )),
            gender=random.choice(["M", "F", "X"]),
            postal_code=random.choice(POSTAL_PREFIXES) + " " + fake.bothify("?#?").upper(),
            registration_date=reg_date,
        )
        patients.append(p)
    session.add_all(patients)
    session.flush()
    return patients


def _age_distribution():
    """Roughly mirrors Canadian primary care demographics."""
    ages = np.arange(2, 90)
    weights = np.ones(len(ages))
    # kids and elderly visit more often
    weights[:10] *= 1.3
    weights[60:] *= 1.5
    # working-age adults are the bulk
    weights[20:60] *= 2.0
    return weights / weights.sum()


def _noshow_probability(
    lead_time_days: int,
    day_of_week: int,
    hour: int,
    appointment_type: str,
    reminders_sent: int,
    is_new_patient: bool,
    patient_age: int,
) -> float:
    """
    Calculate no-show probability based on known risk factors.

    Not trying to be perfect here — just baking in the patterns
    that the ML model should be able to pick up on.
    """
    base = 0.09  # low baseline — risk factors push it up

    # lead time: continuous effect, not just buckets
    # every extra day adds risk, with diminishing returns
    base += min(lead_time_days * 0.014, 0.38)

    # monday mornings and friday afternoons are worse
    if day_of_week == 0 and hour < 12:
        base += 0.12
    elif day_of_week == 4 and hour >= 14:
        base += 0.12
    # mid-week is slightly better
    elif day_of_week in (1, 2, 3):
        base -= 0.03

    # reminders help a lot
    if reminders_sent >= 1:
        base -= 0.15
    if reminders_sent >= 2:
        base -= 0.06

    # new patients no-show more
    if is_new_patient:
        base += 0.15

    # walk-ins almost never no-show (they're already there)
    if appointment_type == "walk-in":
        base -= 0.10
    elif appointment_type == "telehealth":
        base += 0.08

    # young adults (18-30) no-show more, seniors less
    if 18 <= patient_age <= 30:
        base += 0.10
    elif patient_age >= 65:
        base -= 0.06

    # clamp and add noise
    base = np.clip(base, 0.02, 0.70)
    base += np.random.normal(0, 0.01)
    return float(np.clip(base, 0.01, 0.75))


def generate_appointments(session, patients):
    patient_map = {p.patient_id: p for p in patients}
    # track each patient's appointment count so we know who's "new"
    patient_visit_count = {p.patient_id: 0 for p in patients}

    appointments = []
    for i in range(1, NUM_APPOINTMENTS + 1):
        patient = random.choices(patients, k=1)[0]
        appt_date = fake.date_time_between(start_date=DATE_START, end_date=DATE_END)
        # clinic hours: 8am-5pm
        hour = random.choices(
            range(8, 17),
            weights=[1.2, 1.5, 1.3, 1.1, 1.0, 0.9, 1.0, 1.1, 0.8],
            k=1,
        )[0]
        appt_date = appt_date.replace(
            hour=hour,
            minute=random.choice([0, 15, 30, 45]),
            second=0,
        )

        appt_type = random.choices(APPOINTMENT_TYPES, weights=TYPE_WEIGHTS, k=1)[0]
        lead_time = max(0, int(np.random.exponential(scale=10)))
        if appt_type == "walk-in":
            lead_time = 0

        reminders_sent = 0
        if lead_time >= 3:
            reminders_sent = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2], k=1)[0]

        is_new = patient_visit_count[patient.patient_id] < 2
        prob = _noshow_probability(
            lead_time_days=lead_time,
            day_of_week=appt_date.weekday(),
            hour=hour,
            appointment_type=appt_type,
            reminders_sent=reminders_sent,
            is_new_patient=is_new,
            patient_age=patient_map[patient.patient_id].age,
        )

        # determine outcome
        roll = random.random()
        if roll < prob:
            status = "no-show"
        elif roll < prob + 0.05:
            status = "cancelled"
        else:
            status = "showed"

        appt = Appointment(
            appointment_id=i,
            patient_id=patient.patient_id,
            clinic_id=random.choice(CLINIC_IDS),
            scheduled_datetime=appt_date,
            appointment_type=appt_type,
            booked_lead_time_days=lead_time,
            reminders_sent=reminders_sent,
            status=status,
        )
        appointments.append(appt)
        patient_visit_count[patient.patient_id] += 1

    session.add_all(appointments)
    session.flush()
    return appointments


def generate_reminders(session, appointments):
    reminders = []
    rid = 1
    for appt in appointments:
        for j in range(appt.reminders_sent):
            days_before = random.randint(1, min(3, max(1, appt.booked_lead_time_days)))
            sent_at = appt.scheduled_datetime - timedelta(days=days_before)
            r = Reminder(
                reminder_id=rid,
                appointment_id=appt.appointment_id,
                channel=random.choice(["SMS", "email"]),
                sent_at=sent_at,
                opened=random.random() < 0.65,
            )
            reminders.append(r)
            rid += 1
    session.add_all(reminders)
    session.flush()


def main():
    print("Creating tables...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    session = SessionLocal()
    try:
        print(f"Generating {NUM_PATIENTS} patients...")
        patients = generate_patients(session)

        print(f"Generating {NUM_APPOINTMENTS} appointments...")
        appointments = generate_appointments(session, patients)

        print("Generating reminders...")
        generate_reminders(session, appointments)

        session.commit()

        # reset postgres sequences so new inserts don't collide with seeded IDs
        for table_name, id_col in [
            ("patients", "patient_id"),
            ("appointments", "appointment_id"),
            ("reminders", "reminder_id"),
        ]:
            session.execute(text(
                f"SELECT setval(pg_get_serial_sequence('{table_name}', '{id_col}'), "
                f"COALESCE((SELECT MAX({id_col}) FROM {table_name}), 1))"
            ))
        session.commit()

        # quick sanity check
        total = session.execute(text("SELECT COUNT(*) FROM appointments")).scalar()
        noshows = session.execute(
            text("SELECT COUNT(*) FROM appointments WHERE status = 'no-show'")
        ).scalar()
        rate = noshows / total * 100
        print(f"\nDone. {total} appointments, {noshows} no-shows ({rate:.1f}%)")
        if 15 <= rate <= 25:
            print("No-show rate looks realistic.")
        else:
            print(f"Warning: no-show rate {rate:.1f}% is outside expected 15-25% range.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
