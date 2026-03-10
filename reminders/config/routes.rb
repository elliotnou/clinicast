Rails.application.routes.draw do
  get "/health", to: "health#show"

  namespace :api do
    resources :reminders, only: [:index]
    post "appointments/:external_id/reminders", to: "appointments#plan"
    post "patients/:external_id/opt_out", to: "patients#opt_out"
    post "sync", to: "sync#create"
  end

  post "/webhooks/delivery", to: "webhooks#delivery"
end
