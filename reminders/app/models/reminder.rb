class Reminder < ApplicationRecord
  CHANNELS = %w[sms email].freeze
  STATES = %w[pending queued sending sent delivered failed skipped].freeze
  FINISHED_STATES = %w[sent delivered failed skipped].freeze

  belongs_to :appointment
  has_one :patient, through: :appointment

  validates :channel, inclusion: { in: CHANNELS }
  validates :state, inclusion: { in: STATES }
  validates :hours_before,
            numericality: { only_integer: true, greater_than: 0 }
  validates :send_at, presence: true

  scope :due, ->(now = Time.current) { where(state: "pending", send_at: ..now) }
  scope :outstanding, -> { where(state: %w[pending queued]) }

  def finished?
    FINISHED_STATES.include?(state)
  end

  def to_api
    {
      id: id,
      appointment_id: appointment.external_id,
      patient_id: appointment.patient.external_id,
      channel: channel,
      hours_before: hours_before,
      send_at: send_at,
      state: state,
      attempts: attempts,
      skip_reason: skip_reason,
      sent_at: sent_at,
      delivered_at: delivered_at
    }
  end
end
