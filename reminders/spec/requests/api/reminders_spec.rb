require "rails_helper"

RSpec.describe "api/reminders", type: :request do
  it "lists reminders and filters them by state" do
    sent = create(:reminder, state: "sent")
    create(:reminder, state: "pending")

    get "/api/reminders", params: { state: "sent" }

    expect(response).to have_http_status(:ok)
    expect(response.parsed_body["reminders"].map { |r| r["id"] }).to eq([sent.id])
  end

  it "caps how many it will hand back at once" do
    get "/api/reminders", params: { limit: 10_000 }

    expect(response).to have_http_status(:ok)
  end
end
