function analyzeUrl(url) {
  let score = 0;
  const indicators = [];

  if (!url) {
    return {
      score: 0,
      classification: "Unavailable",
      indicators: ["The current URL could not be read."],
    };
  }

  let parsedUrl;

  try {
    parsedUrl = new URL(url);
  } catch {
    return {
      score: 100,
      classification: "Dangerous",
      indicators: ["The value is not a valid URL."],
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

  if (/[.@_-]/.test(parsedUrl.username + parsedUrl.pathname)) {
    score += 5;
    indicators.push("The URL contains unusual separator characters.");
  }

  const suspiciousWords = [
    "login",
    "verify",
    "secure",
    "account",
    "password",
  ];

  const lowerUrl = url.toLowerCase();

  const foundWord = suspiciousWords.find((word) =>
    lowerUrl.includes(word)
  );

  if (foundWord) {
    score += 15;
    indicators.push(
      `The URL contains the suspicious term "${foundWord}".`
    );
  }

  score = Math.min(score, 100);

  return {
    score,
    classification:
      score >= 60
        ? "Dangerous"
        : score >= 30
        ? "Suspicious"
        : "Likely safe",
    indicators,
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "ANALYZE_PAGE") {
    return;
  }

  const urlResult = analyzeUrl(message.url);

  chrome.tabs.sendMessage(
    message.tabId,
    { type: "ANALYZE_DOM" },
    (domResult) => {
      const lastError = chrome.runtime.lastError;

      if (lastError) {
        sendResponse({
          score: urlResult.score,
          classification: urlResult.classification,
          indicators: [
            ...urlResult.indicators,
            "DOM analysis is not available on this page.",
          ],
        });

        return;
      }

      const finalScore = Math.round(
        urlResult.score * 0.6 + domResult.score * 0.4
      );

      sendResponse({
        score: finalScore,
        classification:
          finalScore >= 60
            ? "Dangerous"
            : finalScore >= 30
            ? "Suspicious"
            : "Likely safe",
        indicators: [
          ...urlResult.indicators.map((item) => `URL: ${item}`),
          ...domResult.indicators.map((item) => `DOM: ${item}`),
        ],
      });
    }
  );

  return true;
});