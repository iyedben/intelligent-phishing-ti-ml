"use strict";

const form = document.getElementById("analysis-form");
const urlInput = document.getElementById("url-input");
const analyzeButton = document.getElementById("analyze-button");
const buttonSpinner = document.getElementById("button-spinner");
const buttonLabel = document.getElementById("button-label");
const inputShell = document.getElementById("input-shell");
const scanBar = document.getElementById("scan-bar");
const loading = document.getElementById("loading");
const idleState = document.getElementById("idle-state");
const formError = document.getElementById("form-error");
const results = document.getElementById("analysis-results");
const globalMessage = document.getElementById("global-message");
const historyList = document.getElementById("history-list");
const historyCount = document.getElementById("history-count");
const historyEmpty = document.getElementById("history-empty");
const healthList = document.getElementById("health-list");
const coverageNote = document.getElementById("coverage-note");

const PROVIDER_NAMES = { urlhaus: "URLhaus", phishtank: "PhishTank", openphish: "OpenPhish" };
const PROVIDER_ORDER = ["phishtank", "urlhaus", "openphish"];
const BOOLEAN_FEATURES = { IsHTTPS: 1, IsDomainIP: 1 };
const LENGTH_FEATURES = { URLLength: 1, DomainLength: 1 };
const FEATURE_LABELS = {
  URLLength: "URL length",
  DomainLength: "Domain length",
  IsDomainIP: "Host is an IP address",
  NoOfDegitsInURL: "Digits in URL",
  NoOfEqualsInURL: "Equals signs",
  NoOfQMarkInURL: "Question marks",
  NoOfAmpersandInURL: "Ampersands",
  IsHTTPS: "Uses HTTPS",
};
const ERROR_TEXT = {
  missing_api_key: "No API key configured, so it was not checked.",
  timeout: "The provider did not respond before the timeout.",
  request_error: "The provider could not be reached.",
  invalid_json: "The provider response could not be parsed.",
  invalid_response: "The provider response could not be interpreted.",
  response_too_large: "The provider response exceeded its size limit.",
  empty_or_invalid_feed: "The community feed was empty or unreadable.",
  oversized_feed: "The community feed exceeded its size limit.",
  provider_internal_error: "The provider lookup could not be completed.",
};

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 64;
let scanHistory = [];
let modelApiOnline = null;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function percent(value, digits) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(digits === undefined ? 1 : digits)}%` : "Unavailable";
}

function humanize(value) {
  return String(value || "").split("_").join(" ");
}

function providerName(source) {
  const key = String(source || "").toLowerCase();
  return PROVIDER_NAMES[key] || String(source || "Provider");
}

function errorText(code) {
  if (!code) return "The provider could not be queried.";
  const known = ERROR_TEXT[code];
  if (known) return known;
  if (String(code).indexOf("http_") === 0) return "The provider returned an HTTP error, so its status is unknown.";
  return "The provider could not be queried.";
}

function featureLabel(name) {
  return FEATURE_LABELS[name] || String(name);
}

function featureValueText(name, value) {
  const numeric = Number(value);
  if (BOOLEAN_FEATURES[name]) return numeric === 1 ? "Yes" : "No";
  if (!Number.isFinite(numeric)) return String(value);
  if (LENGTH_FEATURES[name]) return `${numeric} characters`;
  return numeric === 0 ? "None" : String(numeric);
}

/* ---------- sidebar ---------- */

function renderHealth(providerResults) {
  healthList.replaceChildren();
  const rows = [{
    name: "Model API",
    state: modelApiOnline === null ? "Checking" : modelApiOnline ? "Online" : "Unavailable",
    tone: modelApiOnline === null ? "is-idle" : modelApiOnline ? "" : "is-down",
  }];

  const bySource = {};
  (providerResults || []).forEach((item) => { bySource[String(item.source).toLowerCase()] = item; });

  PROVIDER_ORDER.forEach((key) => {
    const found = bySource[key];
    if (!found) {
      rows.push({ name: PROVIDER_NAMES[key], state: "Not checked", tone: "is-idle" });
      return;
    }
    const status = found.lookup_status;
    if (status === "match" || status === "no_match") {
      rows.push({ name: PROVIDER_NAMES[key], state: "Answered", tone: "" });
    } else if (found.error === "missing_api_key") {
      rows.push({ name: PROVIDER_NAMES[key], state: "No API key", tone: "is-idle" });
    } else {
      rows.push({ name: PROVIDER_NAMES[key], state: "Failing", tone: "is-down" });
    }
  });

  rows.forEach((row) => {
    const line = node("div", "health-row");
    const name = node("span", "health-name");
    const dot = node("span", `health-dot ${row.tone}`.trim());
    dot.setAttribute("aria-hidden", "true");
    name.append(dot, node("span", "", row.name));
    line.append(name, node("span", `health-state ${row.tone}`.trim(), row.state));
    healthList.append(line);
  });
}

function renderHistory() {
  historyList.replaceChildren();
  historyCount.textContent = String(scanHistory.length);
  historyEmpty.hidden = scanHistory.length !== 0;

  scanHistory.forEach((entry, index) => {
    const risky = entry.data.ml.prediction === "phishing";
    const item = node("button", `history-item${index === 0 ? " is-active" : ""}`);
    item.type = "button";
    const dot = node("span", `history-dot${risky ? " is-risk" : ""}`);
    dot.setAttribute("aria-hidden", "true");
    const text = node("span", "history-text");
    text.append(
      node("span", "history-url", entry.data.url.replace(/^https?:\/\//, "")),
      node("span", "history-meta", `${risky ? "High risk" : "Low risk"} · ${percent(entry.data.ml.phishing_probability)}`)
    );
    item.append(dot, text);
    item.addEventListener("click", () => {
      urlInput.value = entry.data.url;
      renderResults(entry.data);
      scanHistory = [entry].concat(scanHistory.filter((other) => other !== entry));
      renderHistory();
    });
    historyList.append(item);
  });
}

function rememberScan(data) {
  scanHistory = [{ data: data }].concat(scanHistory.filter((entry) => entry.data.url !== data.url)).slice(0, 6);
  renderHistory();
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    modelApiOnline = response.ok;
  } catch (_error) {
    modelApiOnline = false;
  }
  renderHealth(null);
}

/* ---------- report ---------- */

function renderGauge(ml) {
  const probability = Math.max(0, Math.min(1, Number(ml.phishing_probability) || 0));
  const threshold = Math.max(0, Math.min(1, Number(ml.threshold) || 0));
  const risky = ml.prediction === "phishing";

  const arc = document.getElementById("gauge-arc");
  arc.setAttribute("stroke", risky ? "#F97316" : "#38BDF8");
  arc.setAttribute("stroke-dasharray", `${(GAUGE_CIRCUMFERENCE * Math.max(probability, 0.01)).toFixed(1)} ${GAUGE_CIRCUMFERENCE.toFixed(1)}`);
  document.getElementById("gauge-tick").setAttribute("transform", `rotate(${(threshold * 360 - 180).toFixed(2)} 80 80)`);
  document.getElementById("gauge-svg").setAttribute("aria-label", `Phishing probability ${percent(probability)}`);
  document.getElementById("probability-value").textContent = percent(probability);
  document.getElementById("threshold-caption").textContent = `Tick marks the ${percent(threshold, 0)} decision threshold`;
  document.getElementById("gauge-card").classList.toggle("is-risk", risky);
}

function renderSignals(data) {
  const list = document.getElementById("signal-list");
  list.replaceChildren();

  const providers = Array.isArray(data.threat_intelligence) ? data.threat_intelligence : [];
  const answered = providers.filter((item) => item.lookup_status === "match" || item.lookup_status === "no_match");
  const matched = providers.filter((item) => item.match === true);
  const risky = data.ml.prediction === "phishing";

  let tiValue = "No answer";
  let tiTone = "";
  if (matched.length) {
    tiValue = "Listed";
    tiTone = "tone-orange";
  } else if (answered.length) {
    tiValue = "No match";
    tiTone = "tone-blue";
  }

  const rows = [
    {
      name: "Model",
      note: "Research model, URL structure only",
      value: risky ? "Looks like phishing" : "Looks safe",
      tone: risky ? "tone-orange" : "tone-blue",
    },
    {
      name: "Threat intelligence",
      note: `${answered.length} of ${providers.length} sources answered`,
      value: tiValue,
      tone: tiTone,
    },
    {
      name: "Overall",
      note: `Evidence level: ${data.correlation.evidence_level || "none"}`,
      value: humanize(data.correlation.status),
      tone: risky ? "tone-solid-orange" : "tone-solid-blue",
    },
  ];

  rows.forEach((row) => {
    const line = node("div", "signal");
    const left = node("div", "");
    left.append(node("div", "signal-name", row.name), node("div", "signal-note", row.note));
    line.append(left, node("span", `signal-value ${row.tone}`.trim(), row.value));
    list.append(line);
  });
}

function renderContributions(explanation) {
  const grid = document.getElementById("contributions");
  grid.replaceChildren();

  const items = Array.isArray(explanation.top_contributions) ? explanation.top_contributions : [];
  document.getElementById("no-contributions").hidden = items.length !== 0;
  document.getElementById("split-bar").hidden = items.length === 0;

  let maxAbs = 0;
  let safeSum = 0;
  let riskSum = 0;
  items.forEach((item) => {
    const value = Number(item.shap_value) || 0;
    maxAbs = Math.max(maxAbs, Math.abs(value));
    if (value < 0) safeSum += -value; else riskSum += value;
  });

  const total = safeSum + riskSum;
  const safeShare = total > 0 ? (safeSum / total) * 100 : 50;
  const splitSafe = document.getElementById("split-safe");
  const splitRisk = document.getElementById("split-risk");
  splitSafe.style.width = `${safeShare.toFixed(1)}%`;
  splitRisk.style.width = `${(100 - safeShare).toFixed(1)}%`;
  splitSafe.textContent = `−${safeSum.toFixed(3)} legitimate`;
  splitRisk.textContent = `+${riskSum.toFixed(3)} phishing`;

  items.forEach((item) => {
    const value = Number(item.shap_value) || 0;
    const risky = value > 0;
    const card = node("div", `feature${risky ? " is-risk" : ""}`);

    const top = node("div", "feature-top");
    top.append(
      node("span", "feature-label", featureLabel(item.feature)),
      node("span", "feature-num", `${risky ? "+" : "−"}${Math.abs(value).toFixed(4)}`)
    );

    const track = node("div", "feature-track");
    const fill = node("span", "feature-fill");
    fill.style.width = maxAbs > 0 ? `${Math.max(3, (Math.abs(value) / maxAbs) * 100)}%` : "0%";
    track.append(fill);

    const foot = node("div", "feature-foot");
    const direction = item.direction === "phishing" ? "Toward phishing" : item.direction === "legitimate" ? "Toward legitimate" : "Neutral";
    foot.append(
      node("span", "", featureValueText(item.feature, item.value)),
      node("span", "feature-dir", direction)
    );

    card.append(top, track, foot);
    grid.append(card);
  });

  const method = explanation.method || "SHAP";
  document.getElementById("shap-method").textContent =
    `${method} values for class ${explanation.explained_class}. They show influence on this model's score, not cause.`;
}

function providerPresentation(provider) {
  const status = provider.lookup_status;
  const verdict = provider.verdict;
  if (status === "match" && provider.match === true) {
    if (verdict === "malware_distribution") {
      return ["tone-orange", "Listed (malware)", "Associated with malware distribution. This is not a phishing verdict."];
    }
    if (verdict === "phishing") {
      return ["tone-orange", "Listed", `Reported as phishing at evidence level "${provider.evidence_level}".`];
    }
    return ["tone-orange", "Record found", `Provider returned a record (${humanize(verdict) || "verdict unknown"}).`];
  }
  if (status === "no_match" && provider.match === false) {
    return ["tone-blue", "No match", "Not in this provider's current dataset. That is not proof the link is safe."];
  }
  if (status === "unavailable") return ["", "Unavailable", errorText(provider.error)];
  if (status === "error") return ["", "Lookup error", errorText(provider.error)];
  if (status === "invalid_input") return ["", "Invalid input", "This provider could not check the supplied URL format."];
  return ["", "Unknown", "The provider returned no definitive match or no-match result."];
}

function renderProviders(providers) {
  const body = document.getElementById("provider-list");
  const coverage = document.getElementById("coverage-text");
  body.replaceChildren();

  if (!Array.isArray(providers) || providers.length === 0) {
    const row = node("tr");
    const cell = node("td", "", "No provider results were returned.");
    cell.colSpan = 4;
    row.append(cell);
    body.append(row);
    coverage.textContent = "";
    return;
  }

  providers.forEach((provider) => {
    const [tone, label, detail] = providerPresentation(provider);
    const row = node("tr");
    const chipCell = node("td");
    chipCell.append(node("span", `intel-chip ${tone}`.trim(), label));
    const time = Number(provider.response_time_ms);
    row.append(
      node("td", "intel-source", providerName(provider.source)),
      chipCell,
      node("td", "", detail),
      node("td", "intel-time", Number.isFinite(time) ? `${time.toFixed(0)} ms` : "—")
    );
    body.append(row);
  });

  const answered = providers.filter((item) => item.lookup_status === "match" || item.lookup_status === "no_match").length;
  coverage.textContent = answered === providers.length
    ? "All configured sources answered."
    : `${providers.length - answered} of ${providers.length} sources did not answer, so a missing match is not proof the link is safe.`;
}

function renderResults(data) {
  document.getElementById("result-url").textContent = data.url;

  const risky = data.ml.prediction === "phishing";
  const title = document.getElementById("prediction-value");
  title.textContent = risky ? "High phishing risk" : "Low phishing risk";
  title.classList.toggle("is-risk", risky);
  document.getElementById("correlation-assessment").textContent =
    data.correlation.assessment || "No correlation assessment was returned.";

  renderGauge(data.ml);
  renderSignals(data);
  renderContributions(data.explanation);
  renderProviders(data.threat_intelligence);
  renderHealth(data.threat_intelligence);
  coverageNote.textContent = document.getElementById("coverage-text").textContent;
  document.getElementById("raw-json").textContent = JSON.stringify(data, null, 2);

  globalMessage.hidden = true;
  idleState.hidden = true;
  results.hidden = false;
  results.setAttribute("aria-busy", "false");
}

/* ---------- tabs ---------- */

const PANELS = { explain: "panel-explain", intel: "panel-intel", raw: "panel-raw" };

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((other) => {
      const active = other === tab;
      other.classList.toggle("is-active", active);
      other.setAttribute("aria-selected", active ? "true" : "false");
    });
    Object.keys(PANELS).forEach((key) => {
      document.getElementById(PANELS[key]).hidden = key !== target;
    });
  });
});

/* ---------- form ---------- */

function validateUrl(raw) {
  if (!raw || raw.trim() === "") return "Enter a URL to scan.";
  if (/\s/.test(raw)) return "Remove whitespace and enter a complete URL.";
  try {
    const parsed = new URL(raw);
    if (!["http:", "https:"].includes(parsed.protocol) || !parsed.hostname) {
      return "Use a complete address that starts with http:// or https://";
    }
  } catch (_error) {
    return "Use a complete address that starts with http:// or https://";
  }
  return "";
}

function displayError(message) {
  globalMessage.textContent = message;
  globalMessage.hidden = false;
  results.hidden = true;
  results.setAttribute("aria-busy", "false");
}

function setBusy(busy) {
  analyzeButton.disabled = busy;
  buttonSpinner.hidden = !busy;
  buttonLabel.textContent = busy ? "Scanning…" : "Scan URL";
  inputShell.classList.toggle("is-busy", busy);
  scanBar.hidden = !busy;
  results.setAttribute("aria-busy", busy ? "true" : "false");
}

document.querySelectorAll(".sample").forEach((sample) => {
  sample.addEventListener("click", () => {
    urlInput.value = sample.dataset.url;
    form.requestSubmit();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.hidden = true;
  globalMessage.hidden = true;
  inputShell.classList.remove("is-invalid");

  const submittedUrl = urlInput.value;
  const validationError = validateUrl(submittedUrl);
  if (validationError) {
    formError.textContent = validationError;
    formError.hidden = false;
    inputShell.classList.add("is-invalid");
    urlInput.focus();
    return;
  }

  analyzeButton.disabled = true;
  setBusy(true);
  loading.hidden = false;

  try {
    const response = await fetch("/api/v1/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ url: submittedUrl }),
    });
    if (response.status === 422) {
      formError.textContent = "Invalid URL. Enter a valid HTTP or HTTPS address.";
      formError.hidden = false;
      inputShell.classList.add("is-invalid");
      return;
    }
    if (response.status >= 500) {
      displayError("Analysis failed. Please try again.");
      return;
    }
    if (!response.ok) {
      displayError("The analysis request could not be completed.");
      return;
    }
    const data = await response.json();
    renderResults(data);
    rememberScan(data);
  } catch (_error) {
    displayError("Unable to reach the analysis API.");
  } finally {
    setBusy(false);
    loading.hidden = true;
  }
});

form.addEventListener("reset", () => {
  formError.hidden = true;
  globalMessage.hidden = true;
  results.hidden = true;
  idleState.hidden = false;
  inputShell.classList.remove("is-invalid");
  results.setAttribute("aria-busy", "false");
});

renderHealth(null);
renderHistory();
checkHealth();
