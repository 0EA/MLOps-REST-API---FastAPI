// FastAPI ML Inference Lab -- dashboard client.
// Plain JS, native fetch, same-origin relative URLs only. No predictions are
// ever computed here: every result rendered comes from a real response body
// returned by the FastAPI backend.

(() => {
  "use strict";

  const FEATURE_ORDER = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"];
  const REQUEST_TIMEOUT_MS = 8000;
  const MAX_HISTORY = 20;

  // Real published Iris measurements (classic rows from the dataset), used
  // only to pre-fill inputs. Labeled as sample inputs in the UI, not as
  // guaranteed model output.
  const PRESETS = {
    setosa: { sepal_length_cm: 5.1, sepal_width_cm: 3.5, petal_length_cm: 1.4, petal_width_cm: 0.2 },
    versicolor: { sepal_length_cm: 7.0, sepal_width_cm: 3.2, petal_length_cm: 4.7, petal_width_cm: 1.4 },
    virginica: { sepal_length_cm: 6.3, sepal_width_cm: 3.3, petal_length_cm: 6.0, petal_width_cm: 2.5 },
  };

  // Payload used by "Send invalid request": a negative petal length. Sent
  // verbatim via fetch, bypassing the <input min> constraint entirely since
  // we never call form.reportValidity() on this path.
  const INVALID_PAYLOAD = {
    sepal_length_cm: 5.1,
    sepal_width_cm: 3.5,
    petal_length_cm: -1.4,
    petal_width_cm: 0.2,
  };

  const FIELD_LABELS = {
    sepal_length_cm: "Sepal length",
    sepal_width_cm: "Sepal width",
    petal_length_cm: "Petal length",
    petal_width_cm: "Petal width",
  };

  /** @type {Array<object>} newest first, length capped at MAX_HISTORY */
  let history = [];
  let nextRequestId = 1;
  let latestRequestId = 0; // used to ignore stale in-flight responses
  let selectedHistoryId = null; // null => "showing latest" mode

  // ---------- DOM refs ----------
  const els = {
    refreshStatusBtn: document.getElementById("refresh-status-btn"),
    statusBody: document.getElementById("status-body"),
    chipApi: document.getElementById("chip-api"),
    chipModel: document.getElementById("chip-model"),
    chipVersion: document.getElementById("chip-version"),
    chipAccuracy: document.getElementById("chip-accuracy"),
    chipDataset: document.getElementById("chip-dataset"),

    form: document.getElementById("predict-form"),
    predictBtn: document.getElementById("predict-btn"),
    invalidBtn: document.getElementById("invalid-btn"),
    payloadPreviewBody: document.getElementById("payload-preview-body"),
    resultBody: document.getElementById("result-body"),

    tabBtnInspector: document.getElementById("tab-btn-inspector"),
    tabBtnHistory: document.getElementById("tab-btn-history"),
    tabInspector: document.getElementById("tab-inspector"),
    tabHistory: document.getElementById("tab-history"),
    inspectorBody: document.getElementById("inspector-body"),
    flowSteps: document.getElementById("flow-steps"),

    historyList: document.getElementById("history-list"),
    clearHistoryBtn: document.getElementById("clear-history-btn"),
  };

  const inputs = {};
  FEATURE_ORDER.forEach((name) => { inputs[name] = document.getElementById(name); });

  // ---------- helpers ----------
  function readPayload() {
    const payload = {};
    FEATURE_ORDER.forEach((name) => { payload[name] = Number(inputs[name].value); });
    return payload;
  }

  function updatePayloadPreview() {
    els.payloadPreviewBody.textContent = JSON.stringify(readPayload(), null, 2);
  }

  function nowLabel() {
    const d = new Date();
    return d.toLocaleTimeString([], { hour12: false });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function setButtonsDisabled(disabled) {
    els.predictBtn.disabled = disabled;
    els.invalidBtn.disabled = disabled;
  }

  async function fetchJsonWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal });
      return response;
    } finally {
      clearTimeout(timer);
    }
  }

  // ---------- status panel ----------
  function setChip(el, ok, valueText) {
    el.dataset.ok = ok === null ? "" : String(ok);
    el.querySelector(".chip-value").textContent = valueText;
  }

  async function loadStatus() {
    els.statusBody.dataset.state = "loading";
    setChip(els.chipApi, null, "Checking…");
    setChip(els.chipModel, null, "–");
    setChip(els.chipVersion, null, "–");
    setChip(els.chipAccuracy, null, "–");
    setChip(els.chipDataset, null, "–");

    let health;
    try {
      const resp = await fetchJsonWithTimeout("/health", {}, REQUEST_TIMEOUT_MS);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      health = await resp.json();
    } catch (err) {
      setChip(els.chipApi, false, "Unreachable");
      setChip(els.chipModel, false, "Unknown");
      setChip(els.chipVersion, false, "Unknown");
      setChip(els.chipAccuracy, false, "Unknown");
      setChip(els.chipDataset, false, "Unknown");
      els.statusBody.dataset.state = "error";
      return;
    }

    setChip(els.chipApi, true, "Reachable (200 OK)");
    setChip(els.chipModel, Boolean(health.model_loaded), health.model_loaded ? "Yes" : "No");

    try {
      const resp = await fetchJsonWithTimeout("/model-info", {}, REQUEST_TIMEOUT_MS);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const info = await resp.json();

      setChip(els.chipVersion, true, info.model_version);

      const nTest = info?.dataset?.n_samples_test;
      const accuracyPct = (info.held_out_accuracy * 100).toFixed(1) + "%";
      const accuracyText = Number.isFinite(nTest)
        ? `${accuracyPct} (${Math.round(info.held_out_accuracy * nTest)} / ${nTest} test samples)`
        : accuracyPct;
      setChip(els.chipAccuracy, true, accuracyText);

      const datasetText = `${info?.dataset?.name ?? "unknown"} — features: ${info.feature_order.join(", ")} (${info.feature_units})`;
      setChip(els.chipDataset, true, datasetText);

      els.statusBody.dataset.state = "ready";
    } catch (err) {
      setChip(els.chipVersion, false, "Unavailable");
      setChip(els.chipAccuracy, false, "Unavailable");
      setChip(els.chipDataset, false, "Unavailable");
      els.statusBody.dataset.state = "partial";
    }
  }

  // ---------- prediction result panel ----------
  function renderResultLoading() {
    els.resultBody.dataset.state = "loading";
    els.resultBody.innerHTML = `
      <div class="loading-wrap">
        <div class="spinner" role="status" aria-label="Loading"></div>
        Waiting for response from <code>POST /predict</code>&hellip;
      </div>`;
  }

  function renderResultForEntry(entry, { stale } = { stale: false }) {
    if (entry.outcome === "success") {
      const probs = entry.responseBody.probabilities;
      const species = Object.keys(probs);
      const top = entry.responseBody.predicted_species;
      const rows = species
        .sort((a, b) => probs[b] - probs[a])
        .map((name) => {
          const pct = probs[name] * 100;
          return `
            <div class="prob-row">
              <span class="prob-name">${escapeHtml(name)}</span>
              <span class="prob-track"><span class="prob-fill ${name === top ? "is-top" : ""}" style="width:${pct.toFixed(2)}%"></span></span>
              <span class="prob-pct">${pct.toFixed(1)}%</span>
            </div>`;
        })
        .join("");

      els.resultBody.dataset.state = "success";
      els.resultBody.innerHTML = `
        ${stale ? '<span class="result-stale-badge">From an earlier request &ndash; selected from history</span>' : ""}
        <div class="result-species">${escapeHtml(top)}</div>
        <div class="result-meta">Model version ${escapeHtml(entry.responseBody.model_version)} &middot; HTTP ${entry.status} &middot; round-trip ${entry.durationMs} ms</div>
        ${rows}
        <p class="prob-caption">These are per-request model probabilities from <code>POST /predict</code>, not the held-out test accuracy shown in the status panel above.</p>`;
    } else if (entry.outcome === "validation-error") {
      const details = (entry.responseBody?.detail ?? [])
        .map((d) => {
          const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "request body";
          const label = FIELD_LABELS[field] || field;
          return `<li><strong>${escapeHtml(String(label))}</strong> (<code>${escapeHtml(String(field))}</code>): ${escapeHtml(d.msg ?? "invalid value")}</li>`;
        })
        .join("");
      els.resultBody.dataset.state = "validation-error";
      els.resultBody.innerHTML = `
        ${stale ? '<span class="result-stale-badge">From an earlier request &ndash; selected from history</span>' : ""}
        <div class="result-banner">HTTP 422 &middot; Validation rejected this request</div>
        <p class="result-sub">FastAPI/Pydantic rejected the request before it reached the model. Field-level reasons:</p>
        <ul class="result-detail-list">${details || "<li>See response body in the HTTP inspector.</li>"}</ul>`;
    } else {
      els.resultBody.dataset.state = "network-error";
      els.resultBody.innerHTML = `
        ${stale ? '<span class="result-stale-badge">From an earlier request &ndash; selected from history</span>' : ""}
        <div class="result-banner">Request failed</div>
        <p class="result-sub">${escapeHtml(entry.errorMessage || "Could not get a response from the API.")}</p>`;
    }
  }

  // ---------- HTTP inspector ----------
  function statusPillClass(entry) {
    if (entry.outcome === "success") return "ok";
    if (entry.outcome === "validation-error") return "warn";
    return "err";
  }

  function renderInspectorForEntry(entry) {
    const pillClass = statusPillClass(entry);
    const statusLabel = entry.status != null ? `HTTP ${entry.status}` : "No response";
    const requestJson = JSON.stringify(entry.requestBody, null, 2);
    const responseText = entry.responseIsJson
      ? JSON.stringify(entry.responseBody, null, 2)
      : (entry.responseBody ?? "(no response body)");

    els.inspectorBody.innerHTML = `
      <div class="inspector-meta-row">
        <span class="status-pill ${pillClass}">${statusLabel}</span>
        <span class="method-path">${entry.method} ${entry.path}</span>
        <span class="timestamp">at ${entry.time}</span>
        <span class="duration">Round-trip time: <strong>${entry.durationMs != null ? entry.durationMs + " ms" : "n/a"}</strong></span>
      </div>

      ${entry.outcome === "validation-error" ? `
        <div class="validation-explainer">
          Validation rejected the measurements before <code>/predict</code>'s model-inference code ran.
          No prediction was computed for this request.
        </div>` : ""}

      <div class="inspector-columns">
        <div class="code-block-wrap">
          <div class="code-block-label">Outgoing request JSON</div>
          <button class="btn btn-small" data-copy-target="req-json">Copy</button>
          <pre class="code-block" id="req-json">${escapeHtml(requestJson)}</pre>
        </div>

        <div class="code-block-wrap">
          <div class="code-block-label">${entry.responseIsJson ? "Response JSON" : "Response body (raw text)"}</div>
          <button class="btn btn-small" data-copy-target="resp-json">Copy</button>
          <pre class="code-block" id="resp-json">${escapeHtml(String(responseText))}</pre>
        </div>
      </div>
    `;

    els.inspectorBody.querySelectorAll("[data-copy-target]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const text = document.getElementById(btn.dataset.copyTarget).textContent;
        navigator.clipboard?.writeText(text).then(() => {
          const original = btn.textContent;
          btn.textContent = "Copied!";
          setTimeout(() => { btn.textContent = original; }, 1200);
        });
      });
    });

    renderFlowDiagram(entry);
  }

  function renderFlowDiagram(entry) {
    const steps = { browser: "ok", validation: "unknown", model: "unknown", response: "unknown" };
    if (entry.outcome === "success") {
      steps.validation = "ok";
      steps.model = "ok";
      steps.response = "ok";
    } else if (entry.outcome === "validation-error") {
      steps.validation = "rejected";
      steps.model = "skipped";
      steps.response = "ok"; // an error response did come back
    } else {
      // network/timeout error: we only know the browser sent it.
      steps.validation = "unknown";
      steps.model = "unknown";
      steps.response = "unknown";
    }
    els.flowSteps.querySelectorAll(".flow-step").forEach((stepEl) => {
      const key = stepEl.dataset.step;
      stepEl.dataset.status = steps[key];
    });
  }

  // ---------- history ----------
  function renderHistory() {
    if (history.length === 0) {
      els.historyList.innerHTML = '<p class="placeholder">No requests yet.</p>';
      return;
    }
    const header = `
      <div class="history-list-header">
        <span>Time</span><span>Endpoint</span><span>Status</span><span>Round-trip</span><span>Predicted species</span>
      </div>`;
    const rows = history
      .map((entry) => {
        const pillClass = statusPillClass(entry);
        const statusText = entry.status != null ? entry.status : "No response";
        const species = entry.outcome === "success" ? entry.responseBody.predicted_species : "–";
        const selectedClass = entry.id === selectedHistoryId ? "selected" : "";
        return `
          <div class="history-row ${selectedClass}" data-history-id="${entry.id}" role="button" tabindex="0">
            <span class="hr-time">${entry.time}</span>
            <span class="hr-method">${entry.method} ${entry.path}</span>
            <span class="hr-status ${pillClass === "ok" ? "ok" : pillClass === "warn" ? "warn" : "err"}">${statusText}</span>
            <span class="hr-duration">${entry.durationMs != null ? entry.durationMs + " ms" : "n/a"}</span>
            <span class="hr-species">${escapeHtml(String(species))}</span>
          </div>`;
      })
      .join("");
    els.historyList.innerHTML = header + rows;

    els.historyList.querySelectorAll(".history-row").forEach((row) => {
      row.addEventListener("click", () => selectHistoryEntry(Number(row.dataset.historyId)));
      row.addEventListener("keypress", (e) => {
        if (e.key === "Enter") selectHistoryEntry(Number(row.dataset.historyId));
      });
    });
  }

  function selectHistoryEntry(id) {
    const entry = history.find((h) => h.id === id);
    if (!entry) return;
    selectedHistoryId = id;
    const isLatest = history.length > 0 && history[0].id === id;
    renderResultForEntry(entry, { stale: !isLatest });
    renderInspectorForEntry(entry);
    renderHistory();
    activateTab("inspector");
  }

  function clearHistory() {
    history = [];
    selectedHistoryId = null;
    renderHistory();
    els.inspectorBody.innerHTML = '<p class="placeholder">No request selected yet. Send a prediction to see the raw HTTP exchange here.</p>';
    els.flowSteps.querySelectorAll(".flow-step").forEach((stepEl) => { stepEl.dataset.status = ""; });
  }

  // ---------- tabs ----------
  function activateTab(name) {
    const inspectorActive = name === "inspector";
    els.tabBtnInspector.classList.toggle("active", inspectorActive);
    els.tabBtnHistory.classList.toggle("active", !inspectorActive);
    els.tabBtnInspector.setAttribute("aria-selected", String(inspectorActive));
    els.tabBtnHistory.setAttribute("aria-selected", String(!inspectorActive));
    els.tabInspector.classList.toggle("active", inspectorActive);
    els.tabHistory.classList.toggle("active", !inspectorActive);
  }

  // ---------- core predict flow ----------
  async function sendPredict(payload) {
    const requestId = nextRequestId++;
    latestRequestId = requestId;
    selectedHistoryId = null; // new request => back to "latest" mode

    setButtonsDisabled(true);
    renderResultLoading();

    const entry = {
      id: requestId,
      time: nowLabel(),
      method: "POST",
      path: "/predict",
      requestBody: payload,
      status: null,
      responseBody: null,
      responseIsJson: true,
      durationMs: null,
      outcome: null, // "success" | "validation-error" | "network-error"
      errorMessage: null,
    };

    const start = performance.now();
    try {
      const resp = await fetchJsonWithTimeout(
        "/predict",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        REQUEST_TIMEOUT_MS,
      );
      const rawText = await resp.text();
      entry.durationMs = Math.round(performance.now() - start);
      entry.status = resp.status;

      try {
        entry.responseBody = rawText ? JSON.parse(rawText) : null;
        entry.responseIsJson = true;
      } catch {
        entry.responseBody = rawText;
        entry.responseIsJson = false;
      }

      entry.outcome = resp.status === 200 ? "success" : "validation-error";
    } catch (err) {
      entry.durationMs = Math.round(performance.now() - start);
      entry.outcome = "network-error";
      if (err.name === "AbortError") {
        entry.errorMessage = `Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s. Is uvicorn still running on this origin?`;
      } else {
        entry.errorMessage = "Could not reach the API. Is the FastAPI server running?";
      }
    }

    // Ignore stale results if a newer request has since started.
    if (requestId !== latestRequestId) return;

    history.unshift(entry);
    if (history.length > MAX_HISTORY) history.length = MAX_HISTORY;

    renderResultForEntry(entry, { stale: false });
    renderInspectorForEntry(entry);
    renderHistory();
    setButtonsDisabled(false);
  }

  // ---------- wiring ----------
  els.refreshStatusBtn.addEventListener("click", loadStatus);

  FEATURE_ORDER.forEach((name) => {
    inputs[name].addEventListener("input", updatePayloadPreview);
  });

  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendPredict(readPayload());
  });

  els.invalidBtn.addEventListener("click", () => {
    FEATURE_ORDER.forEach((name) => { inputs[name].value = INVALID_PAYLOAD[name]; });
    updatePayloadPreview();
    sendPredict({ ...INVALID_PAYLOAD });
  });

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = PRESETS[btn.dataset.preset];
      FEATURE_ORDER.forEach((name) => { inputs[name].value = preset[name]; });
      updatePayloadPreview();
    });
  });

  els.tabBtnInspector.addEventListener("click", () => activateTab("inspector"));
  els.tabBtnHistory.addEventListener("click", () => activateTab("history"));
  els.clearHistoryBtn.addEventListener("click", clearHistory);

  // ---------- init ----------
  updatePayloadPreview();
  loadStatus();
})();
