from django.db import models


class Patient(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    postal_code = models.CharField(max_length=7)
    registration_date = models.DateField()

    class Meta:
        db_table = "patients"

    def __str__(self):
        return f"Patient {self.pk} ({self.gender}, age {self.age})"


class Appointment(models.Model):
    APPOINTMENT_TYPES = [
        ("GP", "GP"),
        ("telehealth", "Telehealth"),
        ("walk-in", "Walk-in"),
    ]
    STATUS_CHOICES = [
        ("showed", "Showed"),
        ("no-show", "No-show"),
        ("cancelled", "Cancelled"),
    ]

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="appointments")
    clinic_id = models.IntegerField()
    scheduled_datetime = models.DateTimeField()
    appointment_type = models.CharField(max_length=20, choices=APPOINTMENT_TYPES)
    booked_lead_time_days = models.IntegerField()
    reminders_sent = models.IntegerField(default=0)
    status = models.CharField(max_length=15, choices=STATUS_CHOICES)
    noshow_probability = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "appointments"

    def __str__(self):
        return f"Appt {self.pk} — {self.appointment_type} on {self.scheduled_datetime:%b %d}"


class Reminder(models.Model):
    CHANNEL_CHOICES = [
        ("SMS", "SMS"),
        ("email", "Email"),
    ]

    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name="reminders")
    channel = models.CharField(max_length=10, choices=CHANNEL_CHOICES)
    sent_at = models.DateTimeField()
    opened = models.BooleanField(default=False)

    class Meta:
        db_table = "reminders"

    def __str__(self):
        return f"Reminder {self.pk} ({self.channel}) for appt {self.appointment_id}"
