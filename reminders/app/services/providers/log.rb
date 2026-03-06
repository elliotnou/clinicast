module Providers
  # Stands in for a vendor. Writes the message to the log and hands back a
  # receipt shaped like the real thing so the delivery path is exercised end
  # to end without anyone being texted.
  class Log
    def initialize(channel)
      @channel = channel
    end

    def deliver(to:, body:)
      raise DeliveryError, "no address for #{@channel}" if to.blank?

      message_id = "log-#{SecureRandom.uuid}"
      Rails.logger.info("[#{@channel}] to=#{to} id=#{message_id} #{body}")
      Receipt.new(message_id: message_id)
    end
  end
end
