const BACKEND_URL = "http://127.0.0.1:8000/analyze";

function isAnalyzableUrl(url) {
  return (
    typeof url === "string" &&
    (url.startsWith("http://") ||
      url.startsWith("https://"))
  );
}

function analyzeUrl(url) {

  let score = 0;
  const indicators = [];

  if (!url) {
    return {
      score: 0,
      indicators: ["The current URL could not be read."]
    };
  }

  let parsedUrl;

  try {
    parsedUrl = new URL(url);
  } catch {
    return {
      score: 100,
      indicators: ["The value is not a valid URL."]
    };
  }

  if (parsedUrl.protocol !== "https:") {
    score += 10;
    indicators.push("The page does not use HTTPS.");
  }

  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(parsedUrl.hostname)) {
    score += 25;
    indicators.push("The hostname is an IP address.");
  }

  if (url.length > 100) {
    score += 10;
    indicators.push("The URL is unusually long.");
  }

  if (parsedUrl.hostname.split(".").length > 3) {
    score += 10;
    indicators.push("The hostname contains many subdomains.");
  }

  const suspiciousWords = [
    "login",
    "verify",
    "secure",
    "account",
    "password"
  ];

  const lowerUrl = url.toLowerCase();

  const foundWord = suspiciousWords.find((word) =>
    lowerUrl.includes(word)
  );

  if (foundWord) {
    score += 15;
    indicators.push(
      'The URL contains the suspicious term "' + foundWord + '".'
    );
  }

  return {
    score: Math.min(score, 100),
    indicators
  };
}

chrome.runtime.onMessage.addListener(
  (message, sender, sendResponse) => {
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
      async (domResult) => {
        const lastError = chrome.runtime.lastError;

        const domData = lastError
          ? {
              score: 0,
              indicators: []
            }
          : domResult;

        const urlData = analyzeUrl(message.url);

        try {
          const response = await fetch(BACKEND_URL, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              url: message.url,
              url_analysis: urlData,
              dom: domData
            })
          });

          if (!response.ok) {
            throw new Error(
              "Backend returned status " + response.status
            );
          }

          const result = await response.json();

          sendResponse(result);
        } catch (error) {
          sendResponse({
            score: null,
            classification: "Unavailable",
            indicators: [
              ...urlData.indicators.map(
                (item) => "URL: " + item
              ),
              "The backend could not be reached.",
              ...domData.indicators.map(
                (item) => "DOM: " + item
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