class CreatePatients < ActiveRecord::Migration[8.0]
  def change
    create_table :patients do |t|
      t.integer :external_id, null: false
      t.string :phone
      t.string :email
      t.string :time_zone, null: false, default: "America/Toronto"
      t.boolean :sms_opt_in, null: false, default: true
      t.boolean :email_opt_in, null: false, default: true
      t.integer :quiet_hours_start, null: false, default: 21
      t.integer :quiet_hours_end, null: false, default: 8

      t.timestamps
    end

    add_index :patients, :external_id, unique: true
  end
end
