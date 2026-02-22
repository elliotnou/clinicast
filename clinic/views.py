from datetime import datetime

import joblib
import numpy as np
from django.conf import settings
from django.db.models import Count, Q, F
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Appointment, Patient, Reminder
from .serializers import (
    HighRiskAppointmentSerializer,
    PredictRequestSerializer,
    PredictResponseSerializer,
    ReminderRequestSerializer,
    ReminderResponseSerializer,
)

# load model once at module level
_model_bundle = None


def _get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        try:
            _model_bundle = joblib.load(settings.MODEL_PATH)
        except FileNotFoundError:
            pass
    return _model_bundle


def _get_patient_noshow_rate(patient_id):
    from django.db.models import Value, FloatField
    qs = Appointment.objects.filter(
        patient_id=patient_id,
        status__in=["showed", "no-show"],
    )
    total = qs.count()
    if total == 0:
        return 0.20
    noshows = qs.filter(status="no-show").count()
    return noshows / total


def _encode_appointment_type(appt_type):
    bundle = _get_model_bundle()
    if bundle and "label_encoder" in bundle:
        le = bundle["label_encoder"]
        return int(le.transform([appt_type])[0])
    mapping = {"GP": 0, "telehealth": 1, "walk-in": 2}
    return mapping.get(appt_type, 0)


def _age_to_group(age):
    if age <= 17:
        return 0
    elif age <= 30:
        return 1
    elif age <= 50:
        return 2
    elif age <= 65:
        return 3
    return 4


class PredictView(APIView):
    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        bundle = _get_model_bundle()
        if bundle is None:
            return Response(
                {"detail": "Model not loaded"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            patient = Patient.objects.get(pk=data["patient_id"])
        except Patient.DoesNotExist:
            return Response(
                {"detail": f"Patient {data['patient_id']} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        dt = data["scheduled_datetime"]
        noshow_rate = _get_patient_noshow_rate(data["patient_id"])

        features = np.array([[
            data["booked_lead_time_days"],
            dt.weekday(),
            dt.hour,
            _encode_appointment_type(data["appointment_type"]),
            noshow_rate,
            data["reminders_sent"],
            _age_to_group(patient.age),
        ]])

        model = bundle["model"]
        prob = float(model.predict_proba(features)[0][1])

        if prob >= settings.HIGH_RISK_THRESHOLD:
            risk = "high"
        elif prob >= 0.3:
            risk = "medium"
        else:
            risk = "low"

        result = {
            "appointment_type": data["appointment_type"],
            "noshow_probability": round(prob, 4),
            "risk_level": risk,
        }
        return Response(PredictResponseSerializer(result).data)


class HighRiskListView(APIView):
    def get(self, request):
        appointments = (
            Appointment.objects
            .filter(
                noshow_probability__gte=settings.HIGH_RISK_THRESHOLD,
                scheduled_datetime__gte=datetime.now(),
            )
            .exclude(status="cancelled")
            .select_related("patient")
            .order_by("scheduled_datetime")
        )
        serializer = HighRiskAppointmentSerializer(appointments, many=True)
        return Response(serializer.data)


class TriggerReminderView(APIView):
    def post(self, request):
        serializer = ReminderRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            appt = Appointment.objects.get(pk=data["appointment_id"])
        except Appointment.DoesNotExist:
            return Response(
                {"detail": "Appointment not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        reminder = Reminder.objects.create(
            appointment=appt,
            channel=data["channel"],
            sent_at=datetime.now(),
            opened=False,
        )

        appt.reminders_sent = (appt.reminders_sent or 0) + 1
        appt.save(update_fields=["reminders_sent"])

        result = {
            "pk": reminder.pk,
            "appointment_id": reminder.appointment_id,
            "channel": reminder.channel,
            "sent_at": reminder.sent_at,
            "new_reminder_count": appt.reminders_sent,
        }
        return Response(ReminderResponseSerializer(result).data)


class HealthView(APIView):
    def get(self, request):
        bundle = _get_model_bundle()
        return Response({
            "status": "ok",
            "model_loaded": bundle is not None,
        })


def dashboard(request):
    total = Appointment.objects.count()
    noshows = Appointment.objects.filter(status="no-show").count()
    noshow_rate = (noshows / total * 100) if total > 0 else 0

    high_risk_count = Appointment.objects.filter(
        noshow_probability__gte=settings.HIGH_RISK_THRESHOLD
    ).count()

    rf_auc = 0.0
    try:
        meta = joblib.load("models/model_metadata.joblib")
        rf_auc = meta.get("rf_auc", 0.0)
    except Exception:
        pass

    # no-show rate by appointment type
    by_type = []
    for appt_type in ["GP", "telehealth", "walk-in"]:
        qs = Appointment.objects.filter(
            appointment_type=appt_type,
            status__in=["showed", "no-show"],
        )
        type_total = qs.count()
        if type_total > 0:
            type_noshows = qs.filter(status="no-show").count()
            rate = type_noshows / type_total * 100
            by_type.append({
                "type": appt_type,
                "rate": rate,
                "bar_width": int(rate / 40 * 100),
            })
    by_type.sort(key=lambda x: -x["rate"])

    # feature importance
    feature_importance = []
    max_importance = 0.01
    bundle = _get_model_bundle()
    if bundle:
        model = bundle["model"]
        feature_names = bundle.get("features", [])
        display_names = {
            "lead_time_days": "Lead Time",
            "day_of_week": "Day of Week",
            "hour_of_day": "Hour",
            "appointment_type_encoded": "Appt Type",
            "patient_historical_noshow_rate": "Patient History",
            "reminders_sent": "Reminders",
            "patient_age_group": "Age Group",
        }
        importances = model.feature_importances_
        max_importance = max(importances)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            feature_importance.append({
                "name": display_names.get(name, name),
                "importance": float(imp),
                "importance_pct": float(imp) * 100,
                "bar_width": int(imp * 100 / max_importance),
            })

    high_risk_qs = (
        Appointment.objects
        .filter(noshow_probability__gte=settings.HIGH_RISK_THRESHOLD)
        .order_by("-noshow_probability")[:25]
    )
    high_risk = []
    for appt in high_risk_qs:
        appt.noshow_pct = int(appt.noshow_probability * 100)
        high_risk.append(appt)

    return render(request, "dashboard.html", {
        "stats": {
            "total_appointments": total,
            "noshow_rate": noshow_rate,
            "high_risk_count": high_risk_count,
            "rf_auc": rf_auc,
        },
        "by_type": by_type,
        "feature_importance": feature_importance,
        "max_importance": max_importance,
        "high_risk": high_risk,
    })
