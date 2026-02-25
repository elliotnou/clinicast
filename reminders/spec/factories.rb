FactoryBot.define do
  factory :patient do
    sequence(:external_id) { |n| n }
    sequence(:phone) { |n| format("+1416555%04d", n) }
    sequence(:email) { |n| "patient#{n}@example.test" }
    time_zone { "America/Toronto" }
    sms_opt_in { true }
    email_opt_in { true }
    quiet_hours_start { 21 }
    quiet_hours_end { 8 }
  end

  factory :appointment do
    patient
    sequence(:external_id) { |n| n }
    scheduled_at { 10.days.from_now }
    appointment_type { "GP" }
    noshow_probability { 0.72 }
    risk_level { "high" }
    status { "scheduled" }
  end

  factory :reminder do
    appointment
    channel { "sms" }
    hours_before { 24 }
    send_at { 1.hour.ago }
    state { "pending" }
  end
end
