importScripts("config.js");

const ANALYZE_URL = BACKEND_URL + "/analyze";

const activePrechecks = new Set();


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
    ANALYZE_URL,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        url,
        url_analysis: urlData,
        dom: domData
      }),
      signal: AbortSignal.timeout(BACKEND_TIMEOUT_MS)
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


async function precheckTab(tabId, url) {
  if (isDevelopmentUrl(url)) {
    return;
  }

  const key = `${tabId}:${url}`;

  if (activePrechecks.has(key)) {
    return;
  }

  activePrechecks.add(key);

  try {
    const result = await analyzeWithBackend(
      url,
      {
        score: 0,
        indicators: []
      }
    );

    if (shouldBlockAutomatically(result)) {
      await showWarningOverlay(
        tabId,
        url,
        result
      );
    }
  } catch (error) {
    console.warn(
      "Precheck failed:",
      error
    );
  } finally {
    activePrechecks.delete(key);
  }
}


chrome.tabs.onUpdated.addListener(
  async function (tabId, changeInfo, tab) {
    if (changeInfo.status !== "complete") {
      return;
    }

    const url = tab.url;

    if (!isAnalyzableUrl(url)) {
      return;
    }

    if (isDevelopmentUrl(url)) {
      return;
    }

    const bypassKey =
      `bypass:${tabId}:${url}`;

    const bypassData =
      await chrome.storage.session.get(
        bypassKey
      );

    if (bypassData[bypassKey]) {
      await chrome.storage.session.remove(
        bypassKey
      );

      return;
    }

    await precheckTab(tabId, url);
  }
);


chrome.runtime.onMessage.addListener(
  function (message, sender, sendResponse) {
    if (
      message.type ===
      "CONTINUE_ANYWAY_IN_PAGE"
    ) {
      const tabId = sender.tab
        ? sender.tab.id
        : null;

      if (tabId !== null) {
        const url = message.url;

        const bypassKey =
          `bypass:${tabId}:${url}`;

        chrome.storage.session.set({
          [bypassKey]: true
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

    chrome.tabs.sendMessage(
      message.tabId,
      {
        type: "ANALYZE_DOM"
      },
      async function (domResult) {
        const lastError =
          chrome.runtime.lastError;

        const domData =
          lastError || !domResult
            ? {
                score: 0,
                indicators: []
              }
            : domResult;

        try {
          const result =
            await analyzeWithBackend(
              message.url,
              domData
            );

          sendResponse(result);
        } catch {
          const urlData =
            analyzeUrl(message.url);

          sendResponse({
            score: null,
            classification: "Unavailable",
            indicators: [
              ...urlData.indicators.map(
                function (item) {
                  return "URL: " + item;
                }
              ),
              "The backend could not be reached. If it was not " +
              "used for a while, the server needs up to a minute " +
              "to start, please try again.",
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
          });
        }
      }
    );

    return true;
  }
);