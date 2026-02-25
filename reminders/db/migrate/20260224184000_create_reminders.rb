class CreateReminders < ActiveRecord::Migration[8.0]
  def change
    create_table :reminders do |t|
      t.references :appointment, null: false, foreign_key: true
      t.string :channel, null: false
      t.integer :hours_before, null: false
      t.datetime :send_at, null: false
      t.string :state, null: false, default: "pending"
      t.integer :attempts, null: false, default: 0
      t.string :provider_message_id
      t.string :skip_reason
      t.datetime :sent_at
      t.datetime :delivered_at

      t.timestamps
    end

    # one reminder per appointment, channel and offset. this is what stops a
    # patient getting the same message twice if a sync runs more than once.
    add_index :reminders, %i[appointment_id channel hours_before],
              unique: true, name: "index_reminders_on_appointment_and_offset"
    add_index :reminders, :provider_message_id, unique: true
    add_index :reminders, %i[state send_at]
  end
end
