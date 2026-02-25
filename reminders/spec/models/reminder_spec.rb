require "rails_helper"

RSpec.describe Reminder do
  it { is_expected.to belong_to(:appointment) }
  it { is_expected.to validate_inclusion_of(:channel).in_array(Reminder::CHANNELS) }
  it { is_expected.to validate_inclusion_of(:state).in_array(Reminder::STATES) }

  describe ".due" do
    it "picks up pending reminders whose time has come and nothing else" do
      due = create(:reminder, send_at: 5.minutes.ago, state: "pending")
      create(:reminder, send_at: 1.hour.from_now, state: "pending")
      create(:reminder, send_at: 5.minutes.ago, state: "sent")

      expect(Reminder.due).to contain_exactly(due)
    end
  end

  describe "#finished?" do
    it "is true once the reminder has reached a state it cannot leave" do
      expect(build(:reminder, state: "sent")).to be_finished
      expect(build(:reminder, state: "skipped")).to be_finished
      expect(build(:reminder, state: "pending")).not_to be_finished
    end
  end

  it "will not hold the same channel and offset twice for one appointment" do
    reminder = create(:reminder)
    duplicate = Reminder.new(
      appointment: reminder.appointment,
      channel: reminder.channel,
      hours_before: reminder.hours_before,
      send_at: 1.day.from_now
    )

    expect { duplicate.save!(validate: false) }.to raise_error(ActiveRecord::RecordNotUnique)
  end
end
