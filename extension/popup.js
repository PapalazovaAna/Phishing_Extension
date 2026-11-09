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

const scoreRing =
  document.getElementById("score-ring-value");

const scoreLabel =
  document.getElementById("score-label");

const scoreAdvice =
  document.getElementById("score-advice");

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

const SCORE_RING_RADIUS = 34;

// Short advice under the classification, what the user should do.
const ADVICE = {
  safe: "No signs of phishing were found.",
  suspicious: "Be careful before entering any data.",
  dangerous: "Do not enter any personal data on this page."
};

function getAdvice(riskClass, resultClassification) {
  if (resultClassification === "Not available") {
    return "This type of page cannot be analyzed.";
  }

  if (resultClassification === "Unavailable") {
    return "The analysis server could not be reached.";
  }

  return ADVICE[riskClass] || "";
}

function displayScore(
  score,
  riskClass,
  hasScore,
  resultClassification
) {
  const color =
    getColorForRiskClass(riskClass);

  const circumference =
    2 * Math.PI * SCORE_RING_RADIUS;

  const filled = hasScore
    ? circumference * score / 100
    : 0;

  // The ring fills up with the score.
  scoreRing.style.strokeDasharray =
    `${filled} ${circumference}`;
  scoreRing.style.stroke = color;
  scoreRing.style.opacity = filled > 0 ? "1" : "0";

  scoreValue.textContent = hasScore
    ? String(score)
    : "-";
  scoreValue.style.fill = color;

  classification.style.color = color;

  scoreLabel.textContent = hasScore
    ? `Risk score ${score}/100`
    : "Risk score not available";

  scoreAdvice.textContent =
    getAdvice(riskClass, resultClassification);
}

// Indicators that are strong signs of phishing on their own.
const STRONG_INDICATORS = [
  "The hostname is an IP address.",
  "A form submits data to another origin."
];

// Colour level of an indicator: "good" (green), "warning" (orange),
// "danger" (red) or "neutral" (grey). The level comes from the part of
// the result the indicator belongs to, shown by its prefix.
function getIndicatorLevel(indicator, result) {
  const reputation = result.reputation || {};
  const ml = result.ml || {};

  if (indicator.startsWith("ML: ")) {
    if (typeof ml.score !== "number") {
      return "neutral";
    }

    // A high model score that was overruled by a popular domain.
    if (reputation.status === "popular" && ml.score >= 40) {
      return "warning";
    }

    if (ml.score >= 70) {
      return "danger";
    }

    return ml.score >= 40 ? "warning" : "good";
  }

  if (indicator.startsWith("Reputation: ")) {
    const levels = {
      popular: "good",
      suspicious: "warning",
      malicious: "danger"
    };

    return levels[reputation.status] || "neutral";
  }

  if (
    indicator.startsWith("URL: ") ||
    indicator.startsWith("DOM: ")
  ) {
    const text = indicator.slice(indicator.indexOf(" ") + 1);

    return STRONG_INDICATORS.includes(text) ? "danger" : "warning";
  }

  return "neutral";
}

function displayIndicators(indicators, result) {
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
    item.classList.add(getIndicatorLevel(indicator, result));
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
    hasScore,
    resultClassification
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

  displayIndicators(result.indicators, result);

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