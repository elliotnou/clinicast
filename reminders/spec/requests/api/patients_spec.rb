require "rails_helper"

RSpec.describe "api/patients", type: :request do
  it "opts the patient out and stops anything already planned" do
    patient = create(:patient)
    appointment = create(:appointment, patient: patient)
    reminder = create(:reminder, appointment: appointment, channel: "sms", state: "pending")

    post "/api/patients/#{patient.external_id}/opt_out", params: { channel: "sms" }

    expect(response).to have_http_status(:ok)
    expect(patient.reload.sms_opt_in).to be(false)
    expect(reminder.reload).to have_attributes(state: "skipped", skip_reason: "patient opted out")
  end

  it "leaves the other channel alone" do
    patient = create(:patient)

    post "/api/patients/#{patient.external_id}/opt_out", params: { channel: "sms" }

    expect(patient.reload.email_opt_in).to be(true)
  end

  it "rejects a channel it does not have" do
    patient = create(:patient)

    post "/api/patients/#{patient.external_id}/opt_out", params: { channel: "carrier-pigeon" }

    expect(response).to have_http_status(422)
  end

  it "404s for a patient it does not hold" do
    post "/api/patients/999999/opt_out"

    expect(response).to have_http_status(:not_found)
  end
end
