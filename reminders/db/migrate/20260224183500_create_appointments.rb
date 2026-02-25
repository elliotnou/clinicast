class CreateAppointments < ActiveRecord::Migration[8.0]
  def change
    create_table :appointments do |t|
      t.integer :external_id, null: false
      t.references :patient, null: false, foreign_key: true
      t.datetime :scheduled_at, null: false
      t.string :appointment_type, null: false
      t.float :noshow_probability, null: false, default: 0.0
      t.string :risk_level, null: false, default: "low"
      t.string :status, null: false, default: "scheduled"
      t.datetime :synced_at

      t.timestamps
    end

    add_index :appointments, :external_id, unique: true
    add_index :appointments, :scheduled_at
  end
end
