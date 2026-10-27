"use strict";

const form = document.getElementById("analysis-form");
const urlInput = document.getElementById("url-input");
const analyzeButton = document.getElementById("analyze-button");
const loading = document.getElementById("loading");
const formError = document.getElementById("form-error");
const results = document.getElementById("analysis-results");
const globalMessage = document.getElementById("global-message");
const apiStatus = document.getElementById("api-status");

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function percent(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(2)}%` : "Unavailable";
}

function setApiStatus(online) {
  apiStatus.classList.toggle("online", online);
  apiStatus.classList.toggle("offline", !online);
  apiStatus.lastElementChild.textContent = online ? "API Online" : "API Unavailable";
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    setApiStatus(response.ok);
  } catch (_error) {
    setApiStatus(false);
  }
}

function validateUrl(raw) {
  if (!raw || raw.trim() === "") return "Enter a URL to analyze.";
  if (/\s/.test(raw)) return "Remove whitespace and enter a complete URL.";
  try {
    const parsed = new URL(raw);
    if (!["http:", "https:"].includes(parsed.protocol) || !parsed.hostname) {
      return "Enter a complete HTTP or HTTPS URL.";
    }
  } catch (_error) {
    return "Enter a valid HTTP or HTTPS URL.";
  }
  return "";
}

function displayError(message) {
  globalMessage.textContent = message;
  globalMessage.hidden = false;
  results.hidden = true;
}

function formatProviderName(source) {
  const names = { urlhaus: "URLhaus", phishtank: "PhishTank", openphish: "OpenPhish" };
  return names[String(source).toLowerCase()] || String(source || "Threat intelligence provider");
}

function providerPresentation(provider) {
  const status = provider.lookup_status;
  if (status === "match" && provider.match === true) {
    const verdict = provider.verdict;
    if (verdict === "malware_distribution") {
      return ["match", "Match", "Provider intelligence associates this URL with malware distribution."];
    }
    if (verdict === "phishing") {
      return ["match", "Phishing listing", "This provider reports phishing evidence at its stated evidence level."];
    }
    return ["match", "Match", `Provider returned a record (${verdict || "verdict unknown"}).`];
  }
  if (status === "no_match" && provider.match === false) {
    const isOpenPhish = String(provider.source).toLowerCase() === "openphish" || provider.verdict === "not_listed_in_community_feed";
    return ["no-match", isOpenPhish ? "Not listed in current community feed" : "No match", "Not found in this provider's current dataset. This is not proof that the URL is safe."];
  }
  if (status === "unavailable") {
    return ["unavailable", "Unavailable", providerErrorText(provider.error)];
  }
  if (status === "error") {
    return ["error", "Lookup error", providerErrorText(provider.error)];
  }
  if (status === "invalid_input") {
    return ["unknown", "Invalid input", "This provider could not check the supplied URL format."];
  }
  return ["unknown", "Unknown", "The provider returned no definitive match or no-match result."];
}

function providerErrorText(error) {
  const known = {
    missing_api_key: "Required provider credentials are not configured.",
    timeout: "The provider did not respond before the request timed out.",
    request_error: "The provider could not be reached.",
    http_error: "The provider returned an HTTP error.",
    invalid_response: "The provider response could not be interpreted.",
    feed_unavailable: "The community feed is currently unavailable.",
    provider_internal_error: "The provider lookup could not be completed.",
  };
  if (!error) return "The provider could not be queried. No-match status is unknown.";
  const safeCode = String(error).split("_").slice(0, 6).join("_");
  return `${known[safeCode] || "The provider could not be queried."} No-match status is unknown.`;
}

function renderContribution(item, maxAbs) {
  const direction = ["phishing", "legitimate", "neutral"].includes(item.direction) ? item.direction : "neutral";
  const row = node("div", `contribution ${direction}`);
  const main = node("div", "contribution-main");
  main.append(node("span", "contribution-feature", item.feature));
  main.append(node("span", "contribution-value", `Value: ${item.value}`));
  const bar = node("div", "contribution-bar");
  bar.setAttribute("aria-hidden", "true");
  const fill = node("span");
  const value = Number(item.shap_value);
  fill.style.width = Number.isFinite(value) && maxAbs > 0 ? `${Math.max(3, Math.abs(value) / maxAbs * 100)}%` : "0%";
  bar.append(fill);
  const sign = Number.isFinite(value) ? `${value > 0 ? "+" : ""}${value.toFixed(4)}` : "Unavailable";
  row.append(main, bar, node("span", "contribution-sign", sign));
  const statement = item.statement || (direction === "phishing" ? "Contributed toward phishing in the model." : direction === "legitimate" ? "Contributed toward legitimate in the model." : "Neutral contribution in the model.");
  row.append(node("span", "contribution-statement", statement));
  return row;
}

function renderProviders(providers) {
  const list = document.getElementById("provider-list");
  list.replaceChildren();
  if (!Array.isArray(providers) || providers.length === 0) {
    list.append(node("p", "empty-note", "No provider results were returned."));
    return;
  }
  providers.forEach((provider) => {
    const [state, label, description] = providerPresentation(provider);
    const card = node("article", `provider ${state}`);
    const top = node("div", "provider-top");
    const dot = node("span", "provider-dot");
    dot.setAttribute("aria-hidden", "true");
    top.append(dot, node("span", "provider-name", formatProviderName(provider.source)), node("span", "provider-state", label));
    card.append(top, node("p", "provider-description", description));
    const metadata = [];
    if (provider.evidence_level && provider.evidence_level !== "none") metadata.push(`Evidence: ${provider.evidence_level}`);
    if (provider.confidence !== null && provider.confidence !== undefined) metadata.push(`Confidence: ${provider.confidence}`);
    if (provider.response_time_ms !== null && provider.response_time_ms !== undefined && Number.isFinite(Number(provider.response_time_ms))) metadata.push(`Response: ${Number(provider.response_time_ms).toFixed(0)} ms`);
    if (metadata.length) card.append(node("p", "provider-meta", metadata.join(" · ")));
    list.append(card);
  });
}

function renderResults(data) {
  document.getElementById("result-url").textContent = data.url;
  const prediction = data.ml.prediction === "phishing" ? "PHISHING" : "LEGITIMATE";
  document.getElementById("prediction-value").textContent = prediction;
  const predictionCard = document.getElementById("prediction-card");
  predictionCard.classList.toggle("risk-high", data.ml.prediction === "phishing");
  predictionCard.classList.toggle("risk-low", data.ml.prediction === "legitimate");
  document.getElementById("probability-value").textContent = percent(data.ml.phishing_probability);
  document.getElementById("threshold-value").textContent = percent(data.ml.threshold);
  const probability = Math.max(0, Math.min(1, Number(data.ml.phishing_probability) || 0));
  const threshold = Math.max(0, Math.min(1, Number(data.ml.threshold) || 0));
  document.getElementById("probability-bar").style.width = `${probability * 100}%`;
  document.getElementById("threshold-marker").style.left = `${threshold * 100}%`;

  const correlation = data.correlation;
  document.getElementById("correlation-status").textContent = String(correlation.status || "unknown").replaceAll("_", " ");
  document.getElementById("correlation-assessment").textContent = correlation.assessment || "No correlation assessment was returned.";
  document.getElementById("correlation-level").textContent = correlation.evidence_level || "unknown";
  document.getElementById("disagreement-badge").hidden = !correlation.disagreement;
  const ruleWrap = document.getElementById("correlation-rules-wrap");
  const ruleList = document.getElementById("correlation-rules");
  ruleList.replaceChildren();
  if (Array.isArray(correlation.rules_applied) && correlation.rules_applied.length) {
    correlation.rules_applied.forEach((rule) => ruleList.append(node("li", "", rule)));
    ruleWrap.hidden = false;
  } else ruleWrap.hidden = true;

  const explanation = data.explanation;
  document.getElementById("shap-method").textContent = explanation.method || "SHAP";
  document.getElementById("explained-class").textContent = String(explanation.explained_class);
  const contributionList = document.getElementById("contributions");
  contributionList.replaceChildren();
  const contributions = Array.isArray(explanation.top_contributions) ? explanation.top_contributions : [];
  const maxAbs = Math.max(0, ...contributions.map((item) => Math.abs(Number(item.shap_value) || 0)));
  contributions.forEach((item) => contributionList.append(renderContribution(item, maxAbs)));
  document.getElementById("no-contributions").hidden = contributions.length !== 0;
  renderProviders(data.threat_intelligence);
  globalMessage.hidden = true;
  results.hidden = false;
  results.setAttribute("aria-busy", "false");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.hidden = true;
  globalMessage.hidden = true;
  const submittedUrl = urlInput.value;
  const validationError = validateUrl(submittedUrl);
  if (validationError) {
    formError.textContent = validationError;
    formError.hidden = false;
    urlInput.focus();
    return;
  }
  analyzeButton.disabled = true;
  loading.hidden = false;
  results.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/v1/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ url: submittedUrl }),
    });
    if (response.status === 422) {
      formError.textContent = "Invalid URL. Please enter a valid HTTP or HTTPS URL.";
      formError.hidden = false;
      results.setAttribute("aria-busy", "false");
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
    renderResults(await response.json());
  } catch (_error) {
    displayError("Unable to reach the analysis API.");
  } finally {
    analyzeButton.disabled = false;
    loading.hidden = true;
    if (results.hidden) results.setAttribute("aria-busy", "false");
  }
});

form.addEventListener("reset", () => {
  formError.hidden = true;
  globalMessage.hidden = true;
  results.hidden = true;
  results.setAttribute("aria-busy", "false");
});

checkHealth();
