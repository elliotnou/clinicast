require "rails_helper"

RSpec.describe SyncHighRiskAppointmentsJob do
  let(:row) do
    {
      "id" => 501,
      "patient_id" => 77,
      "scheduled_datetime" => 12.days.from_now.utc.iso8601,
      "appointment_type" => "GP",
      "noshow_probability" => 0.74,
      "reminders_sent" => 0
    }
  end

  def stub_high_risk(rows)
    stub_request(:get, "http://app:8000/api/appointments/high-risk")
      .to_return(status: 200, body: rows.to_json,
                 headers: { "Content-Type" => "application/json" })
  end

  it "mirrors the appointment and plans its reminders" do
    stub_high_risk([row])

    described_class.perform_now

    appointment = Appointment.find_by(external_id: 501)
    expect(appointment.risk_level).to eq("high")
    expect(appointment.patient.external_id).to eq(77)
    expect(appointment.reminders.count).to eq(3)
  end

  it "can run again without duplicating anything" do
    stub_high_risk([row])

    described_class.perform_now
    described_class.perform_now

    expect(Appointment.count).to eq(1)
    expect(Patient.count).to eq(1)
    expect(Reminder.count).to eq(3)
  end

  it "follows the risk level when the score moves" do
    stub_high_risk([row])
    described_class.perform_now

    stub_high_risk([row.merge("noshow_probability" => 0.42)])
    described_class.perform_now

    expect(Appointment.find_by(external_id: 501).risk_level).to eq("medium")
  end

  it "keeps going when one row is unusable" do
    stub_high_risk([row.merge("appointment_type" => nil), row.merge("id" => 502)])

    expect { described_class.perform_now }.not_to raise_error
    expect(Appointment.find_by(external_id: 502)).to be_present
  end
end
