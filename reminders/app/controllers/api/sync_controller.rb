module Api
  class SyncController < ApplicationController
    def create
      SyncHighRiskAppointmentsJob.perform_later

      render json: { status: "queued" }, status: :accepted
    end
  end
end
