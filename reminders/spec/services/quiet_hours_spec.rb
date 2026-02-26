require "rails_helper"

RSpec.describe QuietHours do
  subject(:quiet_hours) do
    described_class.new(time_zone: "America/Toronto", starts_at: 21, ends_at: 8)
  end

  let(:zone) { ActiveSupport::TimeZone["America/Toronto"] }

  describe "#quiet?" do
    it "is false in the afternoon" do
      expect(quiet_hours.quiet?(zone.parse("2026-03-10 14:00"))).to be(false)
    end

    it "is true late in the evening" do
      expect(quiet_hours.quiet?(zone.parse("2026-03-10 22:30"))).to be(true)
    end

    it "is true overnight" do
      expect(quiet_hours.quiet?(zone.parse("2026-03-10 03:00"))).to be(true)
    end

    it "opens up on the hour the window ends" do
      expect(quiet_hours.quiet?(zone.parse("2026-03-10 08:00"))).to be(false)
    end
  end

  describe "#next_allowed" do
    it "leaves an acceptable time alone" do
      time = zone.parse("2026-03-10 14:00")

      expect(quiet_hours.next_allowed(time)).to eq(time)
    end

    it "holds an evening send until the next morning" do
      expect(quiet_hours.next_allowed(zone.parse("2026-03-10 22:30")))
        .to eq(zone.parse("2026-03-11 08:00"))
    end

    it "holds an overnight send until the same morning" do
      expect(quiet_hours.next_allowed(zone.parse("2026-03-11 03:00")))
        .to eq(zone.parse("2026-03-11 08:00"))
    end

    it "handles a window that does not cross midnight" do
      hours = described_class.new(time_zone: "America/Toronto", starts_at: 0, ends_at: 8)

      expect(hours.next_allowed(zone.parse("2026-03-11 03:00")))
        .to eq(zone.parse("2026-03-11 08:00"))
    end

    it "works off the patient's own zone, not the server's" do
      hours = described_class.new(time_zone: "America/Vancouver", starts_at: 21, ends_at: 8)
      vancouver = ActiveSupport::TimeZone["America/Vancouver"]

      expect(hours.next_allowed(vancouver.parse("2026-03-10 23:00")))
        .to eq(vancouver.parse("2026-03-11 08:00"))
    end

    it "falls back to utc when the zone is not one we know" do
      hours = described_class.new(time_zone: "Mars/Olympus", starts_at: 21, ends_at: 8)

      expect(hours.quiet?(Time.utc(2026, 3, 10, 22, 0))).to be(true)
    end
  end
end
