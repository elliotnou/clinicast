# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[8.0].define(version: 2026_02_24_184000) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"

  create_table "appointments", force: :cascade do |t|
    t.integer "external_id", null: false
    t.bigint "patient_id", null: false
    t.datetime "scheduled_at", null: false
    t.string "appointment_type", null: false
    t.float "noshow_probability", default: 0.0, null: false
    t.string "risk_level", default: "low", null: false
    t.string "status", default: "scheduled", null: false
    t.datetime "synced_at"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["external_id"], name: "index_appointments_on_external_id", unique: true
    t.index ["patient_id"], name: "index_appointments_on_patient_id"
    t.index ["scheduled_at"], name: "index_appointments_on_scheduled_at"
  end

  create_table "patients", force: :cascade do |t|
    t.integer "external_id", null: false
    t.string "phone"
    t.string "email"
    t.string "time_zone", default: "America/Toronto", null: false
    t.boolean "sms_opt_in", default: true, null: false
    t.boolean "email_opt_in", default: true, null: false
    t.integer "quiet_hours_start", default: 21, null: false
    t.integer "quiet_hours_end", default: 8, null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["external_id"], name: "index_patients_on_external_id", unique: true
  end

  create_table "reminders", force: :cascade do |t|
    t.bigint "appointment_id", null: false
    t.string "channel", null: false
    t.integer "hours_before", null: false
    t.datetime "send_at", null: false
    t.string "state", default: "pending", null: false
    t.integer "attempts", default: 0, null: false
    t.string "provider_message_id"
    t.string "skip_reason"
    t.datetime "sent_at"
    t.datetime "delivered_at"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["appointment_id", "channel", "hours_before"], name: "index_reminders_on_appointment_and_offset", unique: true
    t.index ["appointment_id"], name: "index_reminders_on_appointment_id"
    t.index ["provider_message_id"], name: "index_reminders_on_provider_message_id", unique: true
    t.index ["state", "send_at"], name: "index_reminders_on_state_and_send_at"
  end

  add_foreign_key "appointments", "patients"
  add_foreign_key "reminders", "appointments"
end
