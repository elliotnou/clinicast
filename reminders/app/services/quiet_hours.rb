# Works out whether a moment falls inside a patient's quiet hours, and if it
# does, when the next acceptable moment is. All of it happens in the patient's
# own zone; the caller stores the result as UTC like everything else.
class QuietHours
  def initialize(time_zone:, starts_at:, ends_at:)
    @zone = ActiveSupport::TimeZone[time_zone.to_s] || ActiveSupport::TimeZone["UTC"]
    @starts_at = starts_at
    @ends_at = ends_at
  end

  def quiet?(time)
    hour = time.in_time_zone(@zone).hour

    if wraps_midnight?
      hour >= @starts_at || hour < @ends_at
    else
      hour >= @starts_at && hour < @ends_at
    end
  end

  def next_allowed(time)
    return time unless quiet?(time)

    local = time.in_time_zone(@zone)
    target = local.change(hour: @ends_at, min: 0, sec: 0)
    # an evening send waits for the morning, an overnight one is already there
    target += 1.day if local.hour >= @ends_at
    target
  end

  private

  def wraps_midnight?
    @starts_at > @ends_at
  end
end
