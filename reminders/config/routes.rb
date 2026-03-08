Rails.application.routes.draw do
  get "/health", to: "health#show"

  post "/webhooks/delivery", to: "webhooks#delivery"
end
