require_relative "boot"

require "rails"
require "active_model/railtie"
require "active_job/railtie"
require "active_record/railtie"
require "action_controller/railtie"
require "action_view/railtie"

Bundler.require(*Rails.groups)

module Reminders
  class Application < Rails::Application
    config.load_defaults 8.0

    config.autoload_lib(ignore: %w[assets tasks])

    config.api_only = true

    # Everything is stored in UTC. Send windows are worked out in each
    # patient's own zone, which lives on the patient record.
    config.time_zone = "UTC"

    config.active_job.queue_adapter = :sidekiq
  end
end
