# Live demo script (4 minutes) — browser-led

Run this after the 6-minute slide talk. Have the dashboard open and ready;
starting the server itself can be part of the demo if time allows, but have
it already running as a fallback if you're short on time.

Preparation, before you start timing:

```powershell
.venv\Scripts\python.exe scripts\train.py
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open Chrome to **http://127.0.0.1:8000/** and leave a second tab ready for
`/docs`.

## 0:00 – 0:40 — Source code, just enough to orient

Open `app/schemas.py` and `app/main.py`'s `predict` function. Point at:
- `IrisMeasurementRequest` — four required fields, `gt=0`, `allow_inf_nan=False`,
  `extra="forbid"`.
- the explicit feature vector construction in `FEATURE_ORDER`, and the
  one-line comment on why `predict` is `def`, not `async def`.

## 0:40 – 1:10 — Dashboard load: status panel

Switch to the dashboard tab (already open, or refresh it now). Point at the
**Service & model status** panel: API reachable, model loaded, model
version, held-out test accuracy with the sample count — all fetched live
from `/health` and `/model-info`, not hard-coded in the page. Click
**Refresh status** once to show it re-fetches live.

## 1:10 – 2:00 — A real prediction

In the **Flower measurements** panel, click the **Setosa-like** preset
(inputs populate, nothing sent yet), expand **Inspect outgoing JSON
payload** briefly to show exactly what will be sent, then click **Predict
species**. Narrate while the result renders: the species name and the
probability bars came back from one real `POST /predict` call — point out
the probabilities sum to ~100% and are labeled as model probabilities, not
the test accuracy shown above.

## 2:00 – 2:45 — Inspect the HTTP exchange

Scroll to the **HTTP inspector** tab (selected by default after a request).
Show: method/path, timestamp, the outgoing JSON, the actual `200` status,
the response JSON, and the **round-trip time in milliseconds** — explicitly
labeled as round-trip time, not model-inference time. Briefly point at the
request-flow diagram (Browser → validation → model → response) and note out
loud that it's a conceptual diagram of the real outcome, not an
instrumented per-stage trace.

## 2:45 – 3:30 — Invalid input / validation

Back in the input panel, click **Send invalid request**. It sends a
negative petal length straight to `/predict`, bypassing the number inputs'
browser-side constraints entirely. Show: the result panel now shows an
amber **HTTP 422** state naming the exact field and reason, and the
inspector shows the full validation error response body. Emphasize this
error state does not get confused with — or quietly show — the previous
successful prediction.

## 3:30 – 3:50 — Swagger UI, briefly

Switch to the `/docs` tab to show the same `/predict` operation, interactive
and auto-generated from the Pydantic schema — the same contract the
dashboard's `fetch()` calls are using.

## 3:50 – 4:00 — History, if time allows

Open the **Requests from this page** tab, click an earlier row, and show the
inspector restores that exact past request/response pair. Mention it's
scoped to this browser tab only, not a server-side log.

End the demo there. If time is tight, cut from the end backward: drop
history first, then shorten the Swagger UI cutaway to a single sentence.

## Fallback if live serving breaks mid-demo

Have `docs/backup-checklist.md`'s recording ready to play instead, or fall
back to `scripts/demo_client.py` in a terminal — it exercises the same
health/model-info/valid/invalid sequence over plain HTTP. Say so plainly and
move on; don't debug live in front of the room.
