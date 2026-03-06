require "rails_helper"

RSpec.describe EnqueueDueRemindersJob do
  it "hands over the reminders that have come due and leaves the rest" do
    due = create(:reminder, send_at: 10.minutes.ago, state: "pending")
    create(:reminder, send_at: 2.hours.from_now, state: "pending")

    expect { described_class.perform_now }
      .to have_enqueued_job(DeliverReminderJob).with(due.id).exactly(:once)

    expect(due.reload.state).to eq("queued")
  end

  it "does not hand the same reminder over twice" do
    create(:reminder, send_at: 10.minutes.ago, state: "pending")
    described_class.perform_now

    expect { described_class.perform_now }.not_to have_enqueued_job(DeliverReminderJob)
  end
end
