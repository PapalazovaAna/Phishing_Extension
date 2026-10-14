const pageUrlElement =
  document.getElementById("page-url");

const loadingMessage =
  document.getElementById("loading-message");

const errorSection =
  document.getElementById("error-section");

const errorMessage =
  document.getElementById("error-message");

const resultSection =
  document.getElementById("result-section");

const scoreValue =
  document.getElementById("score-value");

const scoreProgress =
  document.getElementById("score-progress");

const classification =
  document.getElementById("classification");

const detailsToggle =
  document.getElementById("details-toggle");

const detailsPanel =
  document.getElementById("details-panel");

const urlScore =
  document.getElementById("url-score");

const mlScore =
  document.getElementById("ml-score");

const domScore =
  document.getElementById("dom-score");

const reputationScore =
  document.getElementById("reputation-score");

const indicatorList =
  document.getElementById("indicator-list");

function setLoading(isLoading) {
  loadingMessage.classList.toggle(
    "hidden",
    !isLoading
  );

  if (isLoading) {
    loadingMessage.textContent =
      "Analyzing page...";
  }
}

function getRiskClass(score, resultClassification) {
  if (
    resultClassification === "Not available" ||
    resultClassification === "Unavailable"
  ) {
    return "unavailable";
  }

  if (score >= 70) {
    return "dangerous";
  }

  if (score >= 40) {
    return "suspicious";
  }

  return "safe";
}

function getColorForRiskClass(riskClass) {
  if (riskClass === "dangerous") {
    return "#dc2626";
  }

  if (riskClass === "suspicious") {
    return "#d97706";
  }

  if (riskClass === "unavailable") {
    return "#6b7280";
  }

  return "#16a34a";
}

function displayScore(score, riskClass, hasScore) {
  scoreValue.textContent = hasScore
    ? `${score}/100`
    : "N/A";

  scoreProgress.style.width = hasScore
    ? `${score}%`
    : "0%";

  const color =
    getColorForRiskClass(riskClass);

  scoreValue.style.color = color;
  scoreProgress.style.background = color;
}

function displayIndicators(indicators) {
  indicatorList.innerHTML = "";

  if (!indicators || indicators.length === 0) {
    const item = document.createElement("li");
    item.textContent =
      "No indicators were detected.";

    indicatorList.appendChild(item);
    return;
  }

  for (const indicator of indicators) {
    const item = document.createElement("li");
    item.textContent = indicator;
    indicatorList.appendChild(item);
  }
}

function displayComponentScore(element, value) {
  if (
    value === null ||
    value === undefined
  ) {
    element.textContent = "N/A";
    return;
  }

  element.textContent = `${value}/100`;
}

function displayResult(result) {
  const hasScore =
    typeof result.score === "number";

  const score = hasScore
    ? result.score
    : 0;

  const resultClassification =
    result.classification || "Unavailable";

  const riskClass = getRiskClass(
    score,
    resultClassification
  );

  resultSection.classList.remove("hidden");
  errorSection.classList.add("hidden");

  classification.textContent =
    resultClassification;

  displayScore(
    score,
    riskClass,
    hasScore
  );

  const components =
    result.components || {};

  displayComponentScore(
    urlScore,
    components.url_score
  );

  displayComponentScore(
    mlScore,
    components.ml_score
  );

  displayComponentScore(
    domScore,
    components.dom_score
  );

  displayComponentScore(
    reputationScore,
    components.reputation_score
  );

  displayIndicators(result.indicators);

  detailsPanel.classList.add("hidden");
  detailsToggle.textContent = "See details";
}

function displayError(message) {
  resultSection.classList.add("hidden");
  errorSection.classList.remove("hidden");
  errorMessage.textContent = message;
}

async function analyzeCurrentPage() {
  setLoading(true);

  try {
    const tabs = await chrome.tabs.query({
      active: true,
      currentWindow: true
    });

    const currentTab = tabs[0];

    if (!currentTab || !currentTab.url) {
      throw new Error(
        "The current page URL could not be read."
      );
    }

    pageUrlElement.textContent =
      currentTab.url;

    const result =
      await chrome.runtime.sendMessage({
        type: "ANALYZE_PAGE",
        url: currentTab.url,
        tabId: currentTab.id
      });

    if (!result) {
      throw new Error(
        "The extension did not receive an analysis result."
      );
    }

    displayResult(result);
  } catch (error) {
    displayError(
      error.message ||
        "An unexpected error occurred."
    );
  } finally {
    setLoading(false);
  }
}

detailsToggle.addEventListener(
  "click",
  () => {
    const isHidden =
      detailsPanel.classList.contains("hidden");

    detailsPanel.classList.toggle(
      "hidden",
      !isHidden
    );

    detailsToggle.textContent = isHidden
      ? "Hide details"
      : "See details";
  }
);

document.addEventListener(
  "DOMContentLoaded",
  analyzeCurrentPage
);