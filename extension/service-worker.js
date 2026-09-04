const BACKEND_URL = "http://127.0.0.1:8000/analyze";

chrome.runtime.onMessage.addListener(
  (message, sender, sendResponse) => {
    if (message.type !== "ANALYZE_PAGE") {
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

        try {
          const response = await fetch(BACKEND_URL, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              url: message.url,
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
            score: domData.score,
            classification: "Unavailable",
            indicators: [
              "The backend could not be reached.",
              ...domData.indicators.map(
                (item) => "DOM: " + item
              )
            ]
          });
        }
      }
    );

    return true;
  }
);