function analyzeDom() {
  let score = 0;
  const indicators = [];

  const passwordFields = document.querySelectorAll(
    'input[type="password"]'
  );

  const forms = document.querySelectorAll("form");

  if (passwordFields.length > 0) {
    score += 10;
    indicators.push(
      "The page contains a password input field."
    );
  }

  const currentOrigin = window.location.origin;

  const externalForm = Array.from(forms).some(
    function (form) {
      const action = form.getAttribute("action");

      if (!action) {
        return false;
      }

      try {
        const formUrl = new URL(
          action,
          window.location.href
        );

        return formUrl.origin !== currentOrigin;
      } catch {
        return false;
      }
    }
  );

  if (externalForm) {
    score += 30;
    indicators.push(
      "A form submits data to another origin."
    );
  }

  const visibleText = document.body
    ? document.body.innerText.toLowerCase()
    : "";

  const suspiciousWords = [
    "verify your account",
    "confirm your password"
  ];

  const foundWord = suspiciousWords.find(
    function (word) {
      return visibleText.includes(word);
    }
  );

  if (foundWord) {
    score += 15;
    indicators.push(
      'The page contains the text "' +
      foundWord +
      '".'
    );
  }

  return {
    score: Math.min(score, 100),
    indicators
  };
}


function removeWarningOverlay() {
  const existingHost = document.getElementById(
    "phishing-risk-warning-host"
  );

  if (existingHost) {
    existingHost.remove();
  }
}


function showWarningOverlay(url, result) {
  removeWarningOverlay();

  const host = document.createElement("div");

  host.id = "phishing-risk-warning-host";

  host.style.position = "fixed";
  host.style.inset = "0";
  host.style.zIndex = "2147483647";

  const shadowRoot = host.attachShadow({
    mode: "open"
  });

  const score = typeof result.score === "number"
    ? result.score
    : "N/A";

  const indicators = Array.isArray(
    result.indicators
  )
    ? result.indicators
    : [];

  const indicatorItems =
    indicators.length > 0
      ? indicators
          .map(
            function (indicator) {
              return (
                "<li>" +
                escapeHtml(indicator) +
                "</li>"
              );
            }
          )
          .join("")
      : "<li>No indicators were detected.</li>";

  shadowRoot.innerHTML = `
    <style>
      :host {
        all: initial;
      }

      .backdrop {
        position: fixed;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: rgba(15, 23, 42, 0.42);
        backdrop-filter: blur(7px);
        -webkit-backdrop-filter: blur(7px);
        font-family: Arial, sans-serif;
      }

      .card {
        width: min(390px, calc(100vw - 40px));
        max-height: min(620px, calc(100vh - 40px));
        overflow: auto;
        padding: 24px;
        border: 1px solid #fecaca;
        border-radius: 14px;
        background: #ffffff;
        color: #1f2937;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.28);
      }

      .icon {
        display: flex;
        width: 42px;
        height: 42px;
        align-items: center;
        justify-content: center;
        margin: 0 auto 12px;
        border-radius: 50%;
        background: #dc2626;
        color: #ffffff;
        font-size: 25px;
        font-weight: bold;
      }

      h1 {
        margin: 0;
        color: #991b1b;
        font-size: 20px;
        text-align: center;
      }

      .description {
        margin: 12px 0;
        color: #4b5563;
        font-size: 13px;
        line-height: 1.45;
        text-align: center;
      }

      .url {
        margin: 14px 0;
        padding: 10px;
        overflow-wrap: anywhere;
        border-radius: 7px;
        background: #f3f4f6;
        color: #4b5563;
        font-size: 11px;
        line-height: 1.4;
      }

      .score {
        display: flex;
        justify-content: space-between;
        margin: 14px 0;
        padding: 11px;
        border-radius: 7px;
        background: #fee2e2;
        color: #991b1b;
        font-size: 13px;
      }

      .score strong {
        font-size: 16px;
      }

      h2 {
        margin: 16px 0 8px;
        color: #374151;
        font-size: 14px;
      }

      ul {
        margin: 0;
        padding-left: 20px;
      }

      li {
        margin-bottom: 6px;
        color: #4b5563;
        font-size: 12px;
        line-height: 1.35;
      }

      .actions {
        display: flex;
        gap: 9px;
        margin-top: 20px;
      }

      button {
        flex: 1;
        padding: 10px 8px;
        border-radius: 7px;
        cursor: pointer;
        font-size: 12px;
        font-weight: bold;
      }

      .back {
        border: 0;
        background: #2563eb;
        color: white;
      }

      .continue {
        border: 1px solid #9ca3af;
        background: white;
        color: #4b5563;
      }

      button:hover {
        opacity: 0.88;
      }
    </style>

    <div class="backdrop">
      <section class="card">
        <div class="icon">!</div>

        <h1>Potential phishing website</h1>

        <p class="description">
          This page contains indicators that may be
          associated with phishing.
        </p>

        <div class="url">
          ${escapeHtml(url)}
        </div>

        <div class="score">
          <span>Risk score</span>
          <strong>${escapeHtml(String(score))}/100</strong>
        </div>

        <h2>Detected indicators</h2>

        <ul>
          ${indicatorItems}
        </ul>

        <div class="actions">
          <button class="back" id="go-back">
            Go back
          </button>

          <button class="continue" id="continue">
            Continue anyway
          </button>
        </div>
      </section>
    </div>
  `;

  document.documentElement.appendChild(host);

  shadowRoot
    .getElementById("continue")
    .addEventListener(
      "click",
      function () {
        chrome.runtime.sendMessage({
          type: "CONTINUE_ANYWAY_IN_PAGE",
          url
        });

        removeWarningOverlay();
      }
    );

  shadowRoot
    .getElementById("go-back")
    .addEventListener(
      "click",
      function () {
        chrome.runtime.sendMessage({
          type: "GO_BACK"
        });
      }
    );
}


function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


chrome.runtime.onMessage.addListener(
  function (message, sender, sendResponse) {
    if (message.type === "ANALYZE_DOM") {
      sendResponse(analyzeDom());
      return;
    }

    if (message.type === "SHOW_PHISHING_WARNING") {
      showWarningOverlay(
        message.url,
        message.result
      );

      sendResponse({ success: true });
    }
  }
);