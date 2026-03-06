# Sends one reminder. Everything that would make sending wrong is checked here
# rather than at planning time, so an opt-out or a cancellation that lands
# after the plan was drawn up is still honoured.
class DeliverReminderJob < ApplicationJob
  queue_as :reminders

  retry_on Providers::DeliveryError, wait: :polynomially_longer, attempts: 5 do |job, error|
    Reminder.where(id: job.arguments.first).update_all(
      ["state = ?, skip_reason = ?, updated_at = ?", "failed", error.message.to_s.first(200), Time.current]
    )
  end

  def perform(reminder_id)
    reminder = Reminder.find_by(id: reminder_id)
    return if reminder.nil? || reminder.finished?
    return unless claim(reminder)

    reminder.reload

    reason = skip_reason(reminder)
    if reason
      reminder.update!(state: "skipped", skip_reason: reason)
      return
    end

    deliver(reminder)
  end

  private

  # A retry of this job may pick its own work back up. Another worker that
  # finds the row already sending may not.
  def claim(reminder)
    states = executions > 1 ? %w[pending queued sending] : %w[pending queued]

    rows = Reminder.where(id: reminder.id, state: states).update_all(
      ["state = ?, attempts = attempts + 1, updated_at = ?", "sending", Time.current]
    )
    rows == 1
  end

  def skip_reason(reminder)
    appointment = reminder.appointment
    patient = appointment.patient

    return "appointment cancelled" if appointment.cancelled?
    return "appointment already passed" if appointment.past?
    return "patient opted out of #{reminder.channel}" unless patient.opted_in?(reminder.channel)
    return "no #{reminder.channel} address on file" if patient.address_for(reminder.channel).blank?

    nil
  end

  def deliver(reminder)
    patient = reminder.appointment.patient

    receipt = Providers.for(reminder.channel).deliver(
      to: patient.address_for(reminder.channel),
      body: ReminderMessage.for(reminder)
    )

    reminder.update!(
      state: "sent",
      sent_at: Time.current,
      provider_message_id: receipt.message_id
    )

    report(reminder)
  end

  # reminders_sent feeds back into the model's features, so the scoring service
  # is told. the message has already gone by this point, so a failure here is
  # logged rather than raised: retrying would text the patient twice.
  def report(reminder)
    ScoringClient.new.record_reminder(
      appointment_id: reminder.appointment.external_id,
      channel: reminder.channel
    )
  rescue ScoringClient::Error => e
    Rails.logger.warn("reminder #{reminder.id} sent but not reported back: #{e.message}")
  end
end
