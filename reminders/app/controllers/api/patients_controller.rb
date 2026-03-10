module Api
  class PatientsController < ApplicationController
    def opt_out
      patient = Patient.find_by(external_id: params[:external_id])
      return render(json: { error: "patient not found" }, status: :not_found) if patient.nil?

      channels = Array(params[:channel].presence || Reminder::CHANNELS)
      unknown = channels - Reminder::CHANNELS
      if unknown.any?
        return render(json: { error: "unknown channel #{unknown.join(', ')}" }, status: 422)
      end

      patient.sms_opt_in = false if channels.include?("sms")
      patient.email_opt_in = false if channels.include?("email")
      patient.save!

      # anything already planned on that channel stops here
      cancelled = patient.reminders.outstanding.where(channel: channels).update_all(
        ["state = ?, skip_reason = ?, updated_at = ?", "skipped", "patient opted out", Time.current]
      )

      render json: {
        patient_id: patient.external_id,
        opted_out: channels,
        cancelled_reminders: cancelled
      }
    end
  end
end
