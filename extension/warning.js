const params =
  new URLSearchParams(window.location.search);

const warningId =
  params.get("id");

const blockedUrl =
  document.getElementById("blocked-url");

const riskScore =
  document.getElementById("risk-score");

const indicatorList =
  document.getElementById("indicator-list");

const goBackButton =
  document.getElementById("go-back-button");

const continueButton =
  document.getElementById("continue-button");

async function loadWarning() {
  if (!warningId) {
    return;
  }

  const key = `warning:${warningId}`;

  const stored =
    await chrome.storage.session.get(key);

  const warningData = stored[key];

  if (!warningData) {
    return;
  }

  const result =
    warningData.result || {};

  blockedUrl.textContent =
    warningData.url || "";

  riskScore.textContent =
    typeof result.score === "number"
      ? `${result.score}/100`
      : "N/A";

  const indicators =
    result.indicators || [];

  for (const indicator of indicators) {
    const item = document.createElement("li");
    item.textContent = indicator;
    indicatorList.appendChild(item);
  }
}

continueButton.addEventListener(
  "click",
  async () => {
    const tabs = await chrome.tabs.query({
      active: true,
      currentWindow: true
    });

    const currentTab = tabs[0];

    await chrome.runtime.sendMessage({
      type: "CONTINUE_ANYWAY",
      warningId,
      tabId: currentTab.id
    });
  }
);

goBackButton.addEventListener(
  "click",
  async () => {
    const tabs = await chrome.tabs.query({
      active: true,
      currentWindow: true
    });

    const currentTab = tabs[0];

    await chrome.runtime.sendMessage({
      type: "GO_BACK",
      tabId: currentTab.id
    });
  }
);

loadWarning();