from rest_framework import serializers


class PredictRequestSerializer(serializers.Serializer):
    patient_id = serializers.IntegerField()
    scheduled_datetime = serializers.DateTimeField()
    appointment_type = serializers.ChoiceField(choices=["GP", "telehealth", "walk-in"])
    booked_lead_time_days = serializers.IntegerField(min_value=0)
    reminders_sent = serializers.IntegerField(min_value=0, default=0)


class PredictResponseSerializer(serializers.Serializer):
    appointment_type = serializers.CharField()
    noshow_probability = serializers.FloatField()
    risk_level = serializers.CharField()


class HighRiskAppointmentSerializer(serializers.Serializer):
    id = serializers.IntegerField(source="pk")
    patient_id = serializers.IntegerField(source="patient.pk")
    scheduled_datetime = serializers.DateTimeField()
    appointment_type = serializers.CharField()
    noshow_probability = serializers.FloatField()
    reminders_sent = serializers.IntegerField()


class ReminderRequestSerializer(serializers.Serializer):
    appointment_id = serializers.IntegerField()
    channel = serializers.ChoiceField(choices=["SMS", "email"], default="SMS")


class ReminderResponseSerializer(serializers.Serializer):
    reminder_id = serializers.IntegerField(source="pk")
    appointment_id = serializers.IntegerField()
    channel = serializers.CharField()
    sent_at = serializers.DateTimeField()
    new_reminder_count = serializers.IntegerField()
