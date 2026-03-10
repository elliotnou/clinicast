require "rails_helper"

RSpec.describe "api/appointments", type: :request do
  it "plans reminders for an appointment on demand" do
    appointment = create(:appointment, risk_level: "low", scheduled_at: 5.days.from_now)

    post "/api/appointments/#{appointment.external_id}/reminders"

    expect(response).to have_http_status(:ok)
    expect(response.parsed_body["planned"].size).to eq(1)
  end

  it "404s for an appointment it does not hold" do
    post "/api/appointments/999999/reminders"

    expect(response).to have_http_status(:not_found)
  end
end
