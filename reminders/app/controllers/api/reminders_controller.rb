module Api
  class RemindersController < ApplicationController
    MAX_LIMIT = 200
    DEFAULT_LIMIT = 50

    def index
      scope = Reminder.includes(appointment: :patient).order(:send_at)
      scope = scope.where(state: params[:state]) if params[:state].present?
      scope = scope.where(channel: params[:channel]) if params[:channel].present?

      render json: { reminders: scope.limit(limit).map(&:to_api) }
    end

    private

    def limit
      requested = params.fetch(:limit, DEFAULT_LIMIT).to_i
      requested = DEFAULT_LIMIT if requested < 1
      [requested, MAX_LIMIT].min
    end
  end
end
