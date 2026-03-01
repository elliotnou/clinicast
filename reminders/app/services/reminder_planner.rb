# Turns one appointment into the reminders it should get. Riskier appointments
# get more warning; every rung is pushed out of the patient's quiet hours, and
# anything that would land in the past or after the appointment is dropped.
class ReminderPlanner
  LADDERS = {
    "high" => [
      { channel: "sms",   hours_before: 168 },
      { channel: "email", hours_before: 48 },
      { channel: "sms",   hours_before: 24 }
    ],
    "medium" => [
      { channel: "email", hours_before: 48 },
      { channel: "sms",   hours_before: 24 }
    ],
    "low" => [
      { channel: "sms", hours_before: 24 }
    ]
  }.freeze

  def initialize(appointment, now: Time.current)
    @appointment = appointment
    @now = now
  end

  def call
    return [] if @appointment.cancelled?
    return [] if @appointment.past?(@now)

    ladder.filter_map { |rung| build(rung) }
  end

  private

  def ladder
    LADDERS.fetch(@appointment.risk_level, LADDERS.fetch("low"))
  end

  def build(rung)
    send_at = send_at_for(rung.fetch(:hours_before))
    return nil if send_at.nil?

    reminder = @appointment.reminders.find_or_initialize_by(
      channel: rung.fetch(:channel),
      hours_before: rung.fetch(:hours_before)
    )
    # already planned on an earlier run, leave it alone
    return nil if reminder.persisted?

    reminder.send_at = send_at
    reminder.save!
    reminder
  rescue ActiveRecord::RecordNotUnique
    # another sync got there first
    nil
  end

  def send_at_for(hours_before)
    target = @appointment.scheduled_at - hours_before.hours
    return nil if target <= @now

    shifted = quiet_hours.next_allowed(target)
    return nil if shifted >= @appointment.scheduled_at

    shifted
  end

  def quiet_hours
    @quiet_hours ||= @appointment.patient.quiet_hours
  end
end
