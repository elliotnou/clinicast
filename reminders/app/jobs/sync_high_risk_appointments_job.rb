# Pulls the current high-risk list from the scoring service, mirrors it locally
# and plans reminders for anything new. Runs on a schedule; safe to run again.
class SyncHighRiskAppointmentsJob < ApplicationJob
  queue_as :reminders

  retry_on ScoringClient::Error, wait: :polynomially_longer, attempts: 3

  def perform
    ScoringClient.new.high_risk_appointments.each { |row| sync(row) }
  end

  private

  def sync(row)
    appointment = upsert(row)
    ReminderPlanner.new(appointment).call
  rescue ActiveRecord::RecordInvalid => e
    # one unusable row should not cost us the rest of the batch
    Rails.logger.warn("skipped appointment #{row['id']}: #{e.message}")
  end

  def upsert(row)
    probability = row.fetch("noshow_probability").to_f

    appointment = Appointment.find_or_initialize_by(external_id: row.fetch("id"))
    appointment.assign_attributes(
      patient: patient_for(row.fetch("patient_id")),
      scheduled_at: Time.zone.parse(row.fetch("scheduled_datetime")),
      appointment_type: row.fetch("appointment_type"),
      noshow_probability: probability,
      risk_level: Appointment.risk_level_for(probability),
      synced_at: Time.current
    )
    appointment.save!
    appointment
  end

  # The scoring service holds no phone or email, so a patient starts life here
  # as a shell and the clinic fills the contact details in.
  #
  # find_or_create_by rather than create_or_find_by: the model validates
  # uniqueness, so a create against an existing patient fails validation before
  # it ever reaches the database, and create_or_find_by only rescues the
  # database error.
  def patient_for(external_id)
    Patient.find_or_create_by!(external_id: external_id)
  rescue ActiveRecord::RecordInvalid, ActiveRecord::RecordNotUnique
    Patient.find_by!(external_id: external_id)
  end
end
