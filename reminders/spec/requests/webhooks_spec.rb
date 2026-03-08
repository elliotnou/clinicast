require "rails_helper"

RSpec.describe "delivery webhooks", type: :request do
  let(:reminder) { create(:reminder, state: "sent", provider_message_id: "log-abc") }

  def with_secret(secret)
    allow(ENV).to receive(:[]).and_call_original
    allow(ENV).to receive(:[]).with("REMINDER_WEBHOOK_SECRET").and_return(secret)
  end

  it "marks the reminder delivered" do
    post "/webhooks/delivery", params: { message_id: reminder.provider_message_id, status: "delivered" }

    expect(response).to have_http_status(:ok)
    expect(reminder.reload.state).to eq("delivered")
    expect(reminder.delivered_at).to be_present
  end

  it "treats an undelivered receipt as a failure" do
    post "/webhooks/delivery", params: { message_id: reminder.provider_message_id, status: "undelivered" }

    expect(reminder.reload.state).to eq("failed")
  end

  it "accepts a receipt for a message it no longer holds" do
    post "/webhooks/delivery", params: { message_id: "log-nope", status: "delivered" }

    expect(response).to have_http_status(:ok)
    expect(response.parsed_body["status"]).to eq("ignored")
  end

  it "rejects a status it does not understand" do
    post "/webhooks/delivery", params: { message_id: reminder.provider_message_id, status: "chewed" }

    expect(response).to have_http_status(422)
  end

  it "turns away a badly signed request" do
    with_secret("shhh")

    post "/webhooks/delivery",
         params: { message_id: reminder.provider_message_id, status: "delivered" },
         headers: { "X-Signature" => "wrong" }

    expect(response).to have_http_status(:unauthorized)
    expect(reminder.reload.state).to eq("sent")
  end

  it "lets a correctly signed request through" do
    with_secret("shhh")
    payload = { message_id: reminder.provider_message_id, status: "delivered" }.to_json
    signature = OpenSSL::HMAC.hexdigest("SHA256", "shhh", payload)

    post "/webhooks/delivery", params: payload,
                               headers: { "X-Signature" => signature, "Content-Type" => "application/json" }

    expect(response).to have_http_status(:ok)
    expect(reminder.reload.state).to eq("delivered")
  end
end
