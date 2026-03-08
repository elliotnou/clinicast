class WebhooksController < ApplicationController
  STATUS_MAP = {
    "delivered" => "delivered",
    "failed" => "failed",
    "undelivered" => "failed"
  }.freeze

  def delivery
    return render(json: { error: "bad signature" }, status: :unauthorized) unless verified?

    state = STATUS_MAP[params[:status].to_s]
    return render(json: { error: "unknown status" }, status: 422) if state.nil?

    reminder = Reminder.find_by(provider_message_id: params[:message_id].to_s)
    # vendors resend receipts for messages we no longer hold. accept them
    # quietly instead of making the vendor retry forever.
    return render(json: { status: "ignored" }) if reminder.nil?

    attributes = { state: state }
    attributes[:delivered_at] = Time.current if state == "delivered"
    reminder.update!(attributes)

    render json: { status: "ok", reminder: reminder.to_api }
  end

  private

  def verified?
    secret = ENV["REMINDER_WEBHOOK_SECRET"]
    return true if secret.blank?

    expected = OpenSSL::HMAC.hexdigest("SHA256", secret, request.raw_post)
    ActiveSupport::SecurityUtils.secure_compare(request.headers["X-Signature"].to_s, expected)
  end
end
