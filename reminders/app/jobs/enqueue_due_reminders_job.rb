# Reminders are planned days ahead, so they sit in the table rather than in
# redis. This picks up the ones that have come due and hands them to a worker.
class EnqueueDueRemindersJob < ApplicationJob
  queue_as :reminders

  BATCH_SIZE = 500

  def perform
    Reminder.due.order(:send_at).limit(BATCH_SIZE).each do |reminder|
      # claiming it here means a second run of this job cannot queue it twice
      claimed = Reminder.where(id: reminder.id, state: "pending").update_all(
        ["state = ?, updated_at = ?", "queued", Time.current]
      )
      next unless claimed == 1

      DeliverReminderJob.perform_later(reminder.id)
    end
  end
end
