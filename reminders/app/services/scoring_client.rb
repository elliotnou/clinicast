require "faraday"
require "faraday/retry"

# Thin wrapper over the Django scoring service. It owns the model and the
# appointment records; this service only ever reads risk from it and reports
# back that a reminder went out.
class ScoringClient
  class Error < StandardError; end

  # the scoring service spells its channels differently to ours
  CHANNEL_NAMES = { "sms" => "SMS", "email" => "email" }.freeze

  def initialize(base_url: nil, timeout: nil)
    @base_url = base_url || ENV.fetch("SCORING_SERVICE_URL", "http://app:8000")
    @timeout = (timeout || ENV.fetch("SCORING_SERVICE_TIMEOUT", 5)).to_i
  end

  def high_risk_appointments
    request { connection.get("/api/appointments/high-risk") }
  end

  def record_reminder(appointment_id:, channel:)
    request do
      connection.post("/api/reminders/trigger", {
        appointment_id: appointment_id,
        channel: CHANNEL_NAMES.fetch(channel.to_s, "SMS")
      })
    end
  end

  private

  def connection
    @connection ||= Faraday.new(url: @base_url) do |f|
      f.request :json
      f.request :retry, max: 2, interval: 0.2, backoff_factor: 2,
                        retry_statuses: [502, 503, 504],
                        exceptions: [Faraday::ConnectionFailed, Faraday::TimeoutError]
      f.response :json
      f.options.timeout = @timeout
      f.options.open_timeout = @timeout
    end
  end

  # the network call has to happen inside the rescue, not before it, or a
  # refused connection sails straight past.
  def request
    response = yield
    raise Error, "scoring service returned #{response.status}" unless response.success?

    response.body
  rescue Faraday::Error => e
    raise Error, "scoring service unreachable: #{e.class}"
  end
end
