from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)
    postal_code: Mapped[str] = mapped_column(String(7), nullable=False)
    registration_date: Mapped[date] = mapped_column(Date, nullable=False)

    appointments: Mapped[list["Appointment"]] = relationship(back_populates="patient")


class Appointment(Base):
    __tablename__ = "appointments"

    appointment_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("patients.patient_id"), nullable=False
    )
    clinic_id: Mapped[int] = mapped_column(Integer, nullable=False)
    scheduled_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    appointment_type: Mapped[str] = mapped_column(String(20), nullable=False)
    booked_lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False)
    reminders_sent: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(15), nullable=False)
    noshow_probability: Mapped[float | None] = mapped_column(Float, nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="appointments")
    reminders: Mapped[list["Reminder"]] = relationship(back_populates="appointment")


class Reminder(Base):
    __tablename__ = "reminders"

    reminder_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    appointment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("appointments.appointment_id"), nullable=False
    )
    channel: Mapped[str] = mapped_column(String(10), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    opened: Mapped[bool] = mapped_column(Boolean, default=False)

    appointment: Mapped["Appointment"] = relationship(back_populates="reminders")
