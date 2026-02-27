require "rails_helper"

RSpec.describe ScoringClient do
  subject(:client) { described_class.new(base_url: "http://scoring.test") }

  describe "#high_risk_appointments" do
    it "returns the rows the scoring service reports" do
      body = [{ "id" => 1, "patient_id" => 9, "noshow_probability" => 0.71 }]
      stub_request(:get, "http://scoring.test/api/appointments/high-risk")
        .to_return(status: 200, body: body.to_json,
                   headers: { "Content-Type" => "application/json" })

      expect(client.high_risk_appointments).to eq(body)
    end

    it "raises when the scoring service errors" do
      stub_request(:get, "http://scoring.test/api/appointments/high-risk")
        .to_return(status: 500, body: "boom")

      expect { client.high_risk_appointments }.to raise_error(described_class::Error, /500/)
    end

    it "raises when the scoring service cannot be reached" do
      stub_request(:get, "http://scoring.test/api/appointments/high-risk")
        .to_raise(Faraday::ConnectionFailed)

      expect { client.high_risk_appointments }.to raise_error(described_class::Error, /unreachable/)
    end
  end

  describe "#record_reminder" do
    it "sends the channel name the scoring service expects" do
      request = stub_request(:post, "http://scoring.test/api/reminders/trigger")
        .with(body: { appointment_id: 4, channel: "SMS" })
        .to_return(status: 200, body: "{}", headers: { "Content-Type" => "application/json" })

      client.record_reminder(appointment_id: 4, channel: "sms")

      expect(request).to have_been_requested
    end
  end
end
