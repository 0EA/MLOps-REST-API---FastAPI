# Backup recording checklist

A live demo can fail for reasons that have nothing to do with the code (no
projector network, port conflict, laptop sleep). Record a backup video
beforehand that covers the same ground as `docs/demo-script.md`, so you can
play it instead without missing content.

This repository does not include a pre-made video or screenshots — none are
faked here. Capture them yourself from the actually running dashboard using
the steps below.

## What to capture (same order as the live demo)

1. **Source code**: `app/schemas.py` (`IrisMeasurementRequest`) and the
   `predict` function in `app/main.py`.
2. **Dashboard, status panel**: `http://127.0.0.1:8000/` right after load —
   API reachable, model loaded, model version, held-out accuracy visible.
3. **Dashboard, successful prediction**: a preset selected, the outgoing
   JSON preview expanded, then the result panel showing the predicted
   species and probability bars after clicking **Predict species**.
4. **Dashboard, HTTP inspector**: the same successful request's method,
   status `200`, request/response JSON, and round-trip time.
5. **Dashboard, invalid request**: after clicking **Send invalid request** —
   the amber `422` result panel and the inspector's validation error body.
6. **Dashboard, request history**: the "Requests from this page" tab with
   at least two entries, one row selected.
7. **Swagger UI**: `http://127.0.0.1:8000/docs`, `POST /predict` expanded.

## How to record (Windows, no extra install required)

1. Start the server first so nothing in the recording shows a blank/loading
   screen:
   ```powershell
   .venv\Scripts\python.exe scripts\train.py
   .venv\Scripts\python.exe -m uvicorn app.main:app --reload
   ```
2. Open `http://127.0.0.1:8000/` in Chrome.
3. Press `Win + Alt + R` to start/stop the built-in Xbox Game Bar recorder
   on the active window, or use `Win + G` to open its recording panel if
   you want to pick the window explicitly. No third-party tool is required
   on Windows 10/11.
4. Record items 1–7 above as one continuous take, narrating briefly as in
   `docs/demo-script.md`. Keep it under ~4 minutes to mirror the real time
   budget.
5. Stop the recording (`Win + Alt + R` again). Game Bar saves to
   `Videos\Captures` by default.
6. Rename the file to something identifiable, e.g.
   `iris-api-dashboard-demo-backup.mp4`, and keep it on the presentation
   laptop (not only in the cloud) in case of no network on presentation day.

## If you can't record ahead of time

At minimum, take static screenshots (`Win + Shift + S`) of dashboard items
2, 3, 4, and 5 above while the server is running locally, plus Swagger UI
(item 7). Also keep the terminal output of `scripts/demo_client.py` as a
plain-text fallback artifact:
```powershell
.venv\Scripts\python.exe scripts\demo_client.py > demo_output.txt
```

## Status

Real browser screenshots of a successful prediction (probability bars
filling correctly) and an HTTP 422 validation rejection have already been
captured and confirmed working in Chrome. The full backup recording
covering items 1–7 above has not been captured yet — do this before
presenting, using the steps above.
