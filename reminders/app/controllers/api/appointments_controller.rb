module Api
  class AppointmentsController < ApplicationController
    def plan
      appointment = Appointment.find_by(external_id: params[:external_id])
      return render(json: { error: "appointment not found" }, status: :not_found) if appointment.nil?

      planned = ReminderPlanner.new(appointment).call

      render json: {
        appointment_id: appointment.external_id,
        risk_level: appointment.risk_level,
        planned: planned.map(&:to_api)
      }
    end
  end
end
