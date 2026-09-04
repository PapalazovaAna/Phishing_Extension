const pageUrlElement = document.querySelector("#page-url");
const analyzeButton = document.querySelector("#analyze-button");
const resultElement = document.querySelector("#result");

let currentUrl = "";
let currentTabId = null;

function showResult(result) {
  const indicators = result.indicators.length
    ? `
      <ul>
        ${result.indicators
          .map((item) => `<li>${item}</li>`)
          .join("")}
      </ul>
    `
    : "<p>No basic indicators were detected.</p>";

  resultElement.innerHTML = `
    <strong>Risk score: ${result.score}/100</strong>
    <p>Classification: ${result.classification}</p>
    ${indicators}
  `;

  resultElement.classList.remove("hidden");
}

async function loadCurrentTab() {
  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true,
  });

  currentUrl = tab?.url ?? "";
  currentTabId = tab?.id ?? null;

  pageUrlElement.textContent =
    currentUrl || "URL is not available";
}

analyzeButton.addEventListener("click", async () => {
  if (!currentUrl || currentTabId === null) {
    resultElement.textContent = "The current page cannot be analyzed.";
    resultElement.classList.remove("hidden");
    return;
  }

  analyzeButton.disabled = true;
  resultElement.textContent = "Analyzing...";
  resultElement.classList.remove("hidden");

  try {
    const response = await chrome.runtime.sendMessage({
      type: "ANALYZE_PAGE",
      url: currentUrl,
      tabId: currentTabId,
    });

    showResult(response);
  } catch (error) {
    resultElement.textContent =
      "An error occurred while analyzing the page.";
  }

  analyzeButton.disabled = false;
});

loadCurrentTab();