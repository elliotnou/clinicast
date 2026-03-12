require "sidekiq"

REDIS_URL = ENV.fetch("REDIS_URL", "redis://localhost:6379/0")

Sidekiq.configure_server do |config|
  config.redis = { url: REDIS_URL }

  config.on(:startup) do
    require "sidekiq/cron"
    schedule = Rails.root.join("config/schedule.yml")
    Sidekiq::Cron::Job.load_from_hash!(YAML.load_file(schedule)) if File.exist?(schedule)
  end
end

Sidekiq.configure_client do |config|
  config.redis = { url: REDIS_URL }
end
