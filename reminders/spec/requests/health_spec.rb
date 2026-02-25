require "rails_helper"

RSpec.describe "health", type: :request do
  it "reports on the pieces the service depends on" do
    get "/health"

    expect(response).to have_http_status(:ok)
    expect(response.parsed_body).to include("status" => "ok", "database" => true)
    expect(response.parsed_body).to have_key("redis")
  end
end
