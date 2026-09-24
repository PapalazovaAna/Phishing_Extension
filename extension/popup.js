const pageUrlElement = document.getElementById("page-url");
const analyzeButton = document.getElementById("analyze-button");
const loadingMessage = document.getElementById("loading-message");
const errorSection = document.getElementById("error-section");
const errorMessage = document.getElementById("error-message");
const resultSection = document.getElementById("result-section");
const warningBanner = document.getElementById("warning-banner");
const scoreValue = document.getElementById("score-value");
const scoreProgress = document.getElementById("score-progress");
const classification = document.getElementById("classification");
const urlScore = document.getElementById("url-score");
const mlScore = document.getElementById("ml-score");
const domScore = document.getElementById("dom-score");
const reputationScore = document.getElementById("reputation-score");
const indicatorList = document.getElementById("indicator-list");


function setLoading(isLoading) {
  loadingMessage.classList.toggle("hidden", !isLoading);
  analyzeButton.disabled = isLoading;

  if (isLoading) {
    analyzeButton.textContent = "Analyzing...";
  } else {
    analyzeButton.textContent = "Analyze again";
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


function getWarningText(score, resultClassification) {
  if (
    resultClassification === "Not available" ||
    resultClassification === "Unavailable"
  ) {
    return "Analysis is currently unavailable.";
  }

  if (score >= 70) {
    return "Warning: This page may be a phishing website.";
  }

  if (score >= 40) {
    return "Caution: This page contains suspicious indicators.";
  }

  return "This page appears to be likely safe.";
}


function displayScore(score, riskClass, hasScore = true) {
  scoreValue.textContent = hasScore
    ? `${score}/100`
    : "N/A";

  scoreProgress.style.width = hasScore
    ? `${score}%`
    : "0%";

  scoreValue.style.color =
    getColorForRiskClass(riskClass);

  scoreProgress.style.background =
    getColorForRiskClass(riskClass);
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


function displayIndicators(indicators) {
  indicatorList.innerHTML = "";

  if (!indicators || indicators.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No indicators were detected.";
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
  if (value === null || value === undefined) {
    element.textContent = "N/A";
    return;
  }

  element.textContent = `${value}/100`;
}


function displayResult(result) {
  const hasScore = typeof result.score === "number";
  const score = hasScore ? result.score : 0;

  const resultClassification =
    result.classification || "Unavailable";

  const riskClass = getRiskClass(
    score,
    resultClassification
  );

  resultSection.classList.remove("hidden");
  errorSection.classList.add("hidden");

  warningBanner.className = `warning-banner ${riskClass}`;
  warningBanner.textContent = getWarningText(
    score,
    resultClassification
  );
  warningBanner.classList.remove("hidden");

  classification.textContent = resultClassification;

  displayScore(score, riskClass, hasScore);

  const components = result.components || {};

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
}


function displayError(message) {
  resultSection.classList.add("hidden");
  errorSection.classList.remove("hidden");
  errorMessage.textContent = message;
}


async function analyzeCurrentPage() {
  setLoading(true);
  errorSection.classList.add("hidden");

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

    pageUrlElement.textContent = currentTab.url;

    const result = await chrome.runtime.sendMessage({
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
      error.message || "An unexpected error occurred."
    );
  } finally {
    setLoading(false);
  }
}


analyzeButton.addEventListener(
  "click",
  analyzeCurrentPage
);


document.addEventListener(
  "DOMContentLoaded",
  analyzeCurrentPage
);