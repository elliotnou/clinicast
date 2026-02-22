from django.contrib import admin

from .models import Appointment, Patient, Reminder


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ["pk", "age", "gender", "postal_code", "registration_date"]
    list_filter = ["gender"]
    search_fields = ["postal_code"]


@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = [
        "pk", "patient", "scheduled_datetime", "appointment_type",
        "status", "noshow_probability", "reminders_sent",
    ]
    list_filter = ["appointment_type", "status"]
    raw_id_fields = ["patient"]


@admin.register(Reminder)
class ReminderAdmin(admin.ModelAdmin):
    list_display = ["pk", "appointment", "channel", "sent_at", "opened"]
    list_filter = ["channel", "opened"]
    raw_id_fields = ["appointment"]
