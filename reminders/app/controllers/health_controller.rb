class HealthController < ApplicationController
  def show
    render json: {
      status: "ok",
      database: database_ok?,
      redis: redis_ok?,
      outstanding_reminders: outstanding_reminders
    }
  end

  private

  def database_ok?
    ActiveRecord::Base.connection.select_value("SELECT 1").to_i == 1
  rescue StandardError
    false
  end

  def redis_ok?
    Sidekiq.redis { |conn| conn.call("PING") } == "PONG"
  rescue StandardError
    false
  end

  def outstanding_reminders
    Reminder.outstanding.count
  rescue StandardError
    nil
  end
end
