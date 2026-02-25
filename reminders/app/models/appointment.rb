class Appointment < ApplicationRecord
  RISK_LEVELS = %w[low medium high].freeze

  # matches the thresholds the scoring service uses in clinicast/settings.py
  HIGH_RISK_THRESHOLD = 0.6
  MEDIUM_RISK_THRESHOLD = 0.3

  belongs_to :patient
  has_many :reminders, dependent: :destroy

  validates :external_id, presence: true, uniqueness: true
  validates :scheduled_at, presence: true
  validates :appointment_type, presence: true
  validates :risk_level, inclusion: { in: RISK_LEVELS }

  scope :upcoming, -> { where(scheduled_at: Time.current..) }
  scope :active, -> { where.not(status: "cancelled") }

  def self.risk_level_for(probability)
    probability = probability.to_f
    return "high" if probability >= HIGH_RISK_THRESHOLD
    return "medium" if probability >= MEDIUM_RISK_THRESHOLD

    "low"
  end

  def cancelled?
    status == "cancelled"
  end

  def past?(now = Time.current)
    scheduled_at <= now
  end
end
