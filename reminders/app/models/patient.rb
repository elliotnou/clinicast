class Patient < ApplicationRecord
  has_many :appointments, dependent: :destroy
  has_many :reminders, through: :appointments

  validates :external_id, presence: true, uniqueness: true
  validates :time_zone, presence: true
  validates :quiet_hours_start, :quiet_hours_end,
            numericality: {
              only_integer: true,
              greater_than_or_equal_to: 0,
              less_than_or_equal_to: 23
            }

  def opted_in?(channel)
    case channel.to_s
    when "sms" then sms_opt_in?
    when "email" then email_opt_in?
    else false
    end
  end

  def address_for(channel)
    case channel.to_s
    when "sms" then phone.presence
    when "email" then email.presence
    end
  end

  def quiet_hours
    QuietHours.new(
      time_zone: time_zone,
      starts_at: quiet_hours_start,
      ends_at: quiet_hours_end
    )
  end
end
