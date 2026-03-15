# Reminders

Rails service that actually sends the reminders Clinicast's model asks for.

The Django side owns the model and the appointment records. It can tell you
which bookings are likely to no-show, but that is where it stops. This service
takes that list and runs the outreach: when to send, on which channel, whether
the patient still wants to hear from us, and what happened to the message.

Kept separate for two reasons. The scoring service holds no phone numbers or
email addresses, so contact details never leave this side. And reminder work is
scheduling, retries and delivery receipts, which is a different job from
serving a model.

## How a reminder happens

```
  every 15 min                          every 5 min
  ------------                          -----------
  GET /api/appointments/high-risk  -->  reminders that have come due
  (django)                              |
     |                                  v
     v                                DeliverReminderJob
  mirror patient + appointment          |
     |                                  |-- checks: cancelled? opted out?
     v                                  |           address on file? past?
  ReminderPlanner                       |
  plans the ladder                      v
                                      provider -> receipt
                                        |
                                        v
                                      POST /api/reminders/trigger (django)
                                      so reminders_sent feeds the model
```

Reminders are planned days ahead and live in Postgres rather than in Redis, so
a queue flush loses nothing. The five-minute job is the only thing that hands
work to a worker.

## The ladder

How much warning a booking gets depends on what the model thinks of it.

| risk | reminders |
|---|---|
| high | SMS 7 days out, email 48h, SMS 24h |
| medium | email 48h, SMS 24h |
| low | SMS 24h |

Rungs that would land in the past are dropped, so a booking made two days out
gets the 24h reminder and nothing else.

## Rules it keeps

- **Quiet hours.** Nothing sends between 21:00 and 08:00 in the patient's own
  zone. A send that lands in the window waits for the morning.
- **Opt-out.** Checked at send time, not at planning time, so a patient who
  opts out after the plan was drawn up still gets nothing.
- **One message per rung.** A unique index on (appointment, channel, offset)
  means a sync that runs twice cannot text anyone twice.
- **Retries.** Provider failures back off and retry five times, then the
  reminder is marked failed rather than retried forever.
- **Delivery receipts.** The provider webhook moves a reminder from sent to
  delivered or failed. Receipts for messages we no longer hold are accepted and
  ignored rather than rejected.

One deliberate asymmetry: if the message goes out but telling Django about it
fails, that is logged and swallowed. Retrying would text the patient a second
time, and a wrong `reminders_sent` count is the cheaper of the two problems.

## API

| | |
|---|---|
| `GET /health` | database, redis, outstanding reminder count |
| `GET /api/reminders` | list, filter by `state` and `channel` |
| `POST /api/appointments/:external_id/reminders` | plan reminders now |
| `POST /api/patients/:external_id/opt_out` | opt out, cancels anything planned |
| `POST /api/sync` | run the sync now instead of waiting |
| `POST /webhooks/delivery` | provider delivery receipt |

The webhook checks an HMAC in `X-Signature` when `REMINDER_WEBHOOK_SECRET` is
set, and runs open when it is not.

## Running it

From the repository root:

```bash
docker compose up --build -d
curl localhost:3000/health
```

That brings up Postgres, Redis, the Django scoring service, this service and a
Sidekiq worker. The worker waits for the web container to pass its health check
so migrations are done before any job runs.

## Tests

```bash
docker compose run --rm reminders bundle exec rspec
```

58 examples covering the planner's ladder and quiet-hours arithmetic, the
delivery job's skip and retry paths, the sync being safe to run twice, and the
webhook's signature check.

## Configuration

| variable | default | |
|---|---|---|
| `SCORING_SERVICE_URL` | `http://app:8000` | where Django is |
| `SCORING_SERVICE_TIMEOUT` | `5` | seconds |
| `REDIS_URL` | `redis://localhost:6379/0` | |
| `REMINDER_PROVIDER` | `log` | `log` writes messages to the log instead of sending them |
| `REMINDER_WEBHOOK_SECRET` | unset | enables webhook signature checking |
| `SIDEKIQ_CONCURRENCY` | `5` | |

There is no real SMS or email vendor wired in. `Providers.for` is the seam;
swapping the log provider for Twilio is a change in `app/services/providers/`
and nowhere else.
