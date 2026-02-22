"""
Generate synthetic clinic appointment data and seed the database.

Creates ~10k appointments over 2 years with realistic no-show patterns
modelled after Canadian primary care clinics (~18-22% no-show rate).
"""

import random
from datetime import timedelta

import numpy as np
from django.core.management.base import BaseCommand
from faker import Faker

from clinic.models import Appointment, Patient, Reminder

fake = Faker("en_CA")

NUM_PATIENTS = 1200
NUM_APPOINTMENTS = 10000
CLINIC_IDS = [1, 2, 3]
APPOINTMENT_TYPES = ["GP", "telehealth", "walk-in"]
TYPE_WEIGHTS = [0.55, 0.25, 0.20]

POSTAL_PREFIXES = [
    "M5V", "M4Y", "L5B", "K1A", "N2L", "T2P", "V6B", "R3C", "E1C", "A1B",
]


def _age_distribution():
    ages = np.arange(2, 90)
    weights = np.ones(len(ages))
    weights[:10] *= 1.3
    weights[60:] *= 1.5
    weights[20:60] *= 2.0
    return weights / weights.sum()


def _noshow_probability(
    lead_time_days, day_of_week, hour, appointment_type,
    reminders_sent, is_new_patient, patient_age,
):
    base = 0.09

    base += min(lead_time_days * 0.014, 0.38)

    if day_of_week == 0 and hour < 12:
        base += 0.12
    elif day_of_week == 4 and hour >= 14:
        base += 0.12
    elif day_of_week in (1, 2, 3):
        base -= 0.03

    if reminders_sent >= 1:
        base -= 0.15
    if reminders_sent >= 2:
        base -= 0.06

    if is_new_patient:
        base += 0.15

    if appointment_type == "walk-in":
        base -= 0.10
    elif appointment_type == "telehealth":
        base += 0.08

    if 18 <= patient_age <= 30:
        base += 0.10
    elif patient_age >= 65:
        base -= 0.06

    base = np.clip(base, 0.02, 0.70)
    base += np.random.normal(0, 0.01)
    return float(np.clip(base, 0.01, 0.75))


class Command(BaseCommand):
    help = "Generate synthetic clinic appointment data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--start", default="2022-01-01",
            help="Start date (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--end", default="2023-12-31",
            help="End date (YYYY-MM-DD)",
        )

    def handle(self, *args, **options):
        from datetime import datetime

        date_start = datetime.strptime(options["start"], "%Y-%m-%d")
        date_end = datetime.strptime(options["end"], "%Y-%m-%d")

        Faker.seed(42)
        np.random.seed(42)
        random.seed(42)

        self.stdout.write("Clearing existing data...")
        Reminder.objects.all().delete()
        Appointment.objects.all().delete()
        Patient.objects.all().delete()

        self.stdout.write(f"Generating {NUM_PATIENTS} patients...")
        patients = []
        for _ in range(NUM_PATIENTS):
            p = Patient.objects.create(
                age=int(np.random.choice(range(2, 90), p=_age_distribution())),
                gender=random.choice(["M", "F", "X"]),
                postal_code=random.choice(POSTAL_PREFIXES) + " " + fake.bothify("?#?").upper(),
                registration_date=fake.date_between(
                    start_date=date_start - timedelta(days=365 * 3),
                    end_date=date_end,
                ),
            )
            patients.append(p)

        self.stdout.write(f"Generating {NUM_APPOINTMENTS} appointments...")
        patient_visit_count = {p.pk: 0 for p in patients}
        appointments = []

        for _ in range(NUM_APPOINTMENTS):
            patient = random.choices(patients, k=1)[0]
            appt_date = fake.date_time_between(start_date=date_start, end_date=date_end)
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

            is_new = patient_visit_count[patient.pk] < 2
            prob = _noshow_probability(
                lead_time_days=lead_time,
                day_of_week=appt_date.weekday(),
                hour=hour,
                appointment_type=appt_type,
                reminders_sent=reminders_sent,
                is_new_patient=is_new,
                patient_age=patient.age,
            )

            roll = random.random()
            if roll < prob:
                appt_status = "no-show"
            elif roll < prob + 0.05:
                appt_status = "cancelled"
            else:
                appt_status = "showed"

            appt = Appointment.objects.create(
                patient=patient,
                clinic_id=random.choice(CLINIC_IDS),
                scheduled_datetime=appt_date,
                appointment_type=appt_type,
                booked_lead_time_days=lead_time,
                reminders_sent=reminders_sent,
                status=appt_status,
            )
            appointments.append(appt)
            patient_visit_count[patient.pk] += 1

        self.stdout.write("Generating reminders...")
        for appt in appointments:
            for j in range(appt.reminders_sent):
                days_before = random.randint(1, min(3, max(1, appt.booked_lead_time_days)))
                sent_at = appt.scheduled_datetime - timedelta(days=days_before)
                Reminder.objects.create(
                    appointment=appt,
                    channel=random.choice(["SMS", "email"]),
                    sent_at=sent_at,
                    opened=random.random() < 0.65,
                )

        total = Appointment.objects.count()
        noshows = Appointment.objects.filter(status="no-show").count()
        rate = noshows / total * 100
        self.stdout.write(f"\nDone. {total} appointments, {noshows} no-shows ({rate:.1f}%)")
        if 15 <= rate <= 25:
            self.stdout.write(self.style.SUCCESS("No-show rate looks realistic."))
        else:
            self.stdout.write(
                self.style.WARNING(f"No-show rate {rate:.1f}% is outside expected 15-25% range.")
            )
