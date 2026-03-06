require "rails_helper"

RSpec.describe DeliverReminderJob do
  let(:patient) { create(:patient, phone: "+14165550000", time_zone: "America/Toronto") }
  let(:appointment) { create(:appointment, patient: patient, scheduled_at: 2.days.from_now) }
  let(:reminder) do
    create(:reminder, appointment: appointment, channel: "sms", send_at: 5.minutes.ago)
  end

  let(:failing_provider) do
    instance_double(Providers::Log).tap do |provider|
      allow(provider).to receive(:deliver).and_raise(Providers::DeliveryError, "vendor timeout")
    end
  end

  before do
    stub_request(:post, "http://app:8000/api/reminders/trigger")
      .to_return(status: 200, body: "{}", headers: { "Content-Type" => "application/json" })
  end

  it "sends the reminder and keeps the provider's message id" do
    described_class.perform_now(reminder.id)

    expect(reminder.reload).to have_attributes(state: "sent", attempts: 1)
    expect(reminder.provider_message_id).to be_present
    expect(reminder.sent_at).to be_present
  end

  it "reports the send back to the scoring service" do
    described_class.perform_now(reminder.id)

    expect(
      a_request(:post, "http://app:8000/api/reminders/trigger")
        .with(body: { appointment_id: appointment.external_id, channel: "SMS" })
    ).to have_been_made
  end

  it "does not send a second time if the job runs again" do
    described_class.perform_now(reminder.id)
    first_id = reminder.reload.provider_message_id

    described_class.perform_now(reminder.id)

    expect(reminder.reload).to have_attributes(provider_message_id: first_id, attempts: 1)
  end

  it "skips a reminder whose appointment was cancelled" do
    appointment.update!(status: "cancelled")

    described_class.perform_now(reminder.id)

    expect(reminder.reload).to have_attributes(state: "skipped", skip_reason: "appointment cancelled")
  end

  it "skips a patient who opted out after the plan was drawn up" do
    patient.update!(sms_opt_in: false)

    described_class.perform_now(reminder.id)

    expect(reminder.reload.state).to eq("skipped")
    expect(reminder.skip_reason).to match(/opted out/)
  end

  it "skips when there is no address on file for the channel" do
    patient.update!(phone: nil)

    described_class.perform_now(reminder.id)

    expect(reminder.reload.skip_reason).to match(/no sms address/)
  end

  it "skips an appointment that has already been and gone" do
    appointment.update_columns(scheduled_at: 1.hour.ago)

    described_class.perform_now(reminder.id)

    expect(reminder.reload.skip_reason).to eq("appointment already passed")
  end

  it "still counts as sent when the scoring service is down" do
    stub_request(:post, "http://app:8000/api/reminders/trigger").to_return(status: 500)

    expect { described_class.perform_now(reminder.id) }.not_to raise_error
    expect(reminder.reload.state).to eq("sent")
  end

  it "retries rather than dropping the reminder when the provider fails" do
    allow(Providers).to receive(:for).and_return(failing_provider)

    expect { described_class.perform_now(reminder.id) }.to have_enqueued_job(described_class)
    expect(reminder.reload.state).to eq("sending")
  end

  it "marks the reminder failed once the retries run out" do
    allow(Providers).to receive(:for).and_return(failing_provider)

    perform_enqueued_jobs { described_class.perform_later(reminder.id) }

    expect(reminder.reload.state).to eq("failed")
    expect(reminder.skip_reason).to match(/vendor timeout/)
  end
end
