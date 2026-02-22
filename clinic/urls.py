from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/predict", views.PredictView.as_view(), name="predict"),
    path("api/appointments/high-risk", views.HighRiskListView.as_view(), name="high-risk"),
    path("api/reminders/trigger", views.TriggerReminderView.as_view(), name="trigger-reminder"),
    path("api/health", views.HealthView.as_view(), name="health"),
]
