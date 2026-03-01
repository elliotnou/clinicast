class ReminderMessage
  def self.for(reminder)
    appointment = reminder.appointment
    local = appointment.scheduled_at.in_time_zone(appointment.patient.time_zone)

    "Reminder: your #{appointment.appointment_type} appointment is on " \
      "#{local.strftime('%a %b %-d')} at #{local.strftime('%-l:%M %p')}. " \
      "Reply STOP to opt out of reminders."
  end
end
