require "rails_helper"

RSpec.describe ReminderPlanner do
  let(:zone) { ActiveSupport::TimeZone["America/Toronto"] }
  let(:patient) { create(:patient, time_zone: "America/Toronto") }
  let(:now) { zone.parse("2026-03-01 10:00") }

  def plan(appointment, at: now)
    described_class.new(appointment, now: at).call
  end

  it "plans the whole ladder for a high risk appointment" do
    appointment = create(:appointment, patient: patient, risk_level: "high",
                                       scheduled_at: zone.parse("2026-03-15 14:00"))

    reminders = plan(appointment)

    expect(reminders.map(&:hours_before)).to contain_exactly(168, 48, 24)
    expect(reminders.map(&:channel)).to contain_exactly("sms", "email", "sms")
  end

  it "plans a shorter ladder for a medium risk appointment" do
    appointment = create(:appointment, patient: patient, risk_level: "medium",
                                       scheduled_at: zone.parse("2026-03-15 14:00"))

    expect(plan(appointment).map(&:hours_before)).to contain_exactly(48, 24)
  end

  it "drops the rungs that would already have passed" do
    appointment = create(:appointment, patient: patient, risk_level: "high",
                                       scheduled_at: zone.parse("2026-03-15 14:00"))

    reminders = plan(appointment, at: zone.parse("2026-03-14 10:00"))

    expect(reminders.map(&:hours_before)).to contain_exactly(24)
  end

  it "moves a send that lands in the patient's quiet hours" do
    appointment = create(:appointment, patient: patient, risk_level: "low",
                                       scheduled_at: zone.parse("2026-03-15 22:00"))

    reminders = plan(appointment)

    expect(reminders.size).to eq(1)
    expect(reminders.first.send_at.in_time_zone(zone)).to eq(zone.parse("2026-03-15 08:00"))
  end

  it "does not plan the same reminder twice when the sync runs again" do
    appointment = create(:appointment, patient: patient, risk_level: "high",
                                       scheduled_at: zone.parse("2026-03-15 14:00"))

    plan(appointment)
    second_pass = plan(appointment.reload)

    expect(second_pass).to be_empty
    expect(appointment.reminders.count).to eq(3)
  end

  it "plans nothing for a cancelled appointment" do
    appointment = create(:appointment, patient: patient, status: "cancelled",
                                       scheduled_at: zone.parse("2026-03-15 14:00"))

    expect(plan(appointment)).to be_empty
  end

  it "plans nothing for an appointment that has already happened" do
    appointment = create(:appointment, patient: patient,
                                       scheduled_at: zone.parse("2026-03-15 14:00"))

    expect(plan(appointment, at: zone.parse("2026-03-16 10:00"))).to be_empty
  end
end
