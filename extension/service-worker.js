const BACKEND_URL =
  "http://127.0.0.1:8000/analyze";

// The latest analysis of each tab: tabId -> { url, domScore, result }.
// "result" is a promise, so the popup can wait for an analysis that
// is still running instead of starting a second one.
const analyses = new Map();


function isAnalyzableUrl(url) {
  return (
    typeof url === "string" &&
    (
      url.startsWith("http://") ||
      url.startsWith("https://")
    )
  );
}


function isDevelopmentUrl(url) {
  try {
    const parsedUrl = new URL(url);

    return (
      parsedUrl.hostname === "localhost" ||
      parsedUrl.hostname === "127.0.0.1" ||
      parsedUrl.hostname === "[::1]"
    );
  } catch {
    return false;
  }
}


function analyzeUrl(url) {
  let score = 0;
  const indicators = [];

  let parsedUrl;

  try {
    parsedUrl = new URL(url);
  } catch {
    return {
      score: 100,
      indicators: [
        "The value is not a valid URL."
      ]
    };
  }

  if (parsedUrl.protocol !== "https:") {
    score += 10;
    indicators.push(
      "The page does not use HTTPS."
    );
  }

  if (
    /^\d{1,3}(\.\d{1,3}){3}$/
      .test(parsedUrl.hostname)
  ) {
    score += 25;
    indicators.push(
      "The hostname is an IP address."
    );
  }

  if (url.length > 100) {
    score += 10;
    indicators.push(
      "The URL is unusually long."
    );
  }

  if (
    parsedUrl.hostname.split(".").length > 3
  ) {
    score += 10;
    indicators.push(
      "The hostname contains many subdomains."
    );
  }

  const suspiciousWords = [
    "login",
    "verify",
    "secure",
    "account",
    "password"
  ];

  const lowerUrl = url.toLowerCase();

  const foundWord = suspiciousWords.find(
    function (word) {
      return lowerUrl.includes(word);
    }
  );

  if (foundWord) {
    score += 15;
    indicators.push(
      `The URL contains the suspicious term "${foundWord}".`
    );
  }

  return {
    score: Math.min(score, 100),
    indicators
  };
}


async function analyzeWithBackend(url, domData) {
  const urlData = analyzeUrl(url);

  const response = await fetch(
    BACKEND_URL,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        url,
        url_analysis: urlData,
        dom: domData
      })
    }
  );

  if (!response.ok) {
    throw new Error(
      "Backend returned status " +
      response.status
    );
  }

  return await response.json();
}


function shouldBlockAutomatically(result) {
  // The backend decides whether a page is blocked,
  // see should_block in backend/app/scoring.py.
  return Boolean(result && result.block === true);
}


async function showWarningOverlay(
  tabId,
  url,
  result
) {
  try {
    await chrome.tabs.sendMessage(
      tabId,
      {
        type: "SHOW_PHISHING_WARNING",
        url,
        result
      }
    );
  } catch (error) {
    console.warn(
      "Could not show warning overlay:",
      error
    );
  }
}


async function isBypassed(tabId, url) {
  const bypassKey = `bypass:${tabId}:${url}`;
  const bypassData = await chrome.storage.session.get(bypassKey);

  return Boolean(bypassData[bypassKey]);
}


function emptyDomData() {
  return {
    score: 0,
    indicators: []
  };
}


// Runs the complete analysis of a page once: URL, ML model,
// reputation and page content (DOM). The same result decides the
// warning and is shown in the popup.
function analyzePage(tabId, url, domData) {
  const result = analyzeWithBackend(url, domData).then(
    async function (analysis) {
      if (
        shouldBlockAutomatically(analysis) &&
        !isDevelopmentUrl(url) &&
        !(await isBypassed(tabId, url))
      ) {
        await showWarningOverlay(tabId, url, analysis);
      }

      return analysis;
    }
  );

  const entry = {
    url,
    domScore: domData.score,
    result
  };

  analyses.set(tabId, entry);

  // A failed analysis is not kept, so the popup can try again.
  result.catch(function (error) {
    console.warn("Analysis failed:", error);

    if (analyses.get(tabId) === entry) {
      analyses.delete(tabId);
    }
  });

  return result;
}


function requestDomData(tabId) {
  return new Promise(function (resolve) {
    chrome.tabs.sendMessage(
      tabId,
      {
        type: "ANALYZE_DOM"
      },
      function (domResult) {
        // No content script (for example a PDF or a page that was
        // open before the extension was loaded): analyze without DOM.
        if (chrome.runtime.lastError || !domResult) {
          resolve(emptyDomData());
          return;
        }

        resolve(domResult);
      }
    );
  });
}


function unavailableResult(url, domData) {
  const urlData = analyzeUrl(url);

  return {
    score: null,
    classification: "Unavailable",
    indicators: [
      ...urlData.indicators.map(
        function (item) {
          return "URL: " + item;
        }
      ),
      "The backend could not be reached.",
      ...domData.indicators.map(
        function (item) {
          return "DOM: " + item;
        }
      )
    ],
    components: {
      url_score: urlData.score,
      ml_score: null,
      dom_score: domData.score,
      reputation_score: null
    }
  };
}


// The popup reuses the analysis made when the page was loaded. The
// page content is checked again first: if it changed since (for
// example a login form appeared later), the page is analyzed again.
async function analysisForPopup(tabId, url) {
  const domData = await requestDomData(tabId);
  const entry = analyses.get(tabId);

  if (
    entry &&
    entry.url === url &&
    entry.domScore === domData.score
  ) {
    try {
      return await entry.result;
    } catch {
      // The stored analysis failed, analyze again below.
    }
  }

  try {
    return await analyzePage(tabId, url, domData);
  } catch {
    return unavailableResult(url, domData);
  }
}


chrome.tabs.onRemoved.addListener(function (tabId) {
  analyses.delete(tabId);
});


chrome.runtime.onMessage.addListener(
  function (message, sender, sendResponse) {
    if (message.type === "PAGE_READY") {
      const tabId = sender.tab ? sender.tab.id : null;

      if (
        tabId !== null &&
        isAnalyzableUrl(message.url) &&
        !isDevelopmentUrl(message.url)
      ) {
        analyzePage(tabId, message.url, message.dom || emptyDomData())
          .catch(function () {
            // Already logged in analyzePage.
          });
      }

      return;
    }

    if (
      message.type ===
      "CONTINUE_ANYWAY_IN_PAGE"
    ) {
      const tabId = sender.tab
        ? sender.tab.id
        : null;

      if (tabId !== null) {
        // Remembered for this tab and address until the browser is
        // closed, so the warning is not shown again on reload.
        chrome.storage.session.set({
          [`bypass:${tabId}:${message.url}`]: true
        });
      }

      sendResponse({
        success: true
      });

      return;
    }

    if (message.type === "GO_BACK") {
      const tabId = sender.tab
        ? sender.tab.id
        : null;

      if (tabId !== null) {
        chrome.tabs.goBack(tabId);
      }

      return;
    }

    if (message.type !== "ANALYZE_PAGE") {
      return;
    }

    if (!isAnalyzableUrl(message.url)) {
      sendResponse({
        score: 0,
        classification: "Not available",
        indicators: [
          "This type of browser page cannot be analyzed."
        ],
        components: {
          url_score: 0,
          ml_score: 0,
          dom_score: 0,
          reputation_score: null
        }
      });

      return;
    }

    analysisForPopup(message.tabId, message.url).then(sendResponse);

    return true;
  }
);
