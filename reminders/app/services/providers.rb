# The seam where a real SMS or email vendor would go. Everything upstream only
# knows deliver(to:, body:) and a receipt with a message id, so swapping the
# log provider for a real one is a change in here and nowhere else.
module Providers
  class DeliveryError < StandardError; end

  Receipt = Struct.new(:message_id, keyword_init: true)

  def self.for(channel)
    case ENV.fetch("REMINDER_PROVIDER", "log")
    when "log" then Log.new(channel)
    else raise ArgumentError, "unknown provider #{ENV['REMINDER_PROVIDER']}"
    end
  end
end
