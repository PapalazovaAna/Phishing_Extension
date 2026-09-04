function analyzeDom() {
  let score = 0;
  const indicators = [];

  const passwordFields = document.querySelectorAll(
    'input[type="password"]'
  );

  const forms = document.querySelectorAll("form");
  const iframes = document.querySelectorAll("iframe");

  if (passwordFields.length > 0) {
    score += 10;
    indicators.push(
      "The page contains a password input field."
    );
  }

  if (forms.length > 0) {
    score += 5;
    indicators.push(
      "The page contains " + forms.length + " form(s)."
    );
  }

  const currentOrigin = window.location.origin;

  const externalForm = Array.from(forms).some(function (form) {
    const action = form.getAttribute("action");

    if (!action) {
      return false;
    }

    try {
      const formUrl = new URL(action, window.location.href);

      return formUrl.origin !== currentOrigin;
    } catch {
      return false;
    }
  });

  if (externalForm) {
    score += 30;
    indicators.push(
      "A form submits data to another origin."
    );
  }

  if (iframes.length > 0) {
    score += 10;
    indicators.push(
      "The page contains " + iframes.length + " iframe(s)."
    );
  }

  const visibleText = document.body
    ? document.body.innerText.toLowerCase()
    : "";

  const suspiciousWords = [
    "verify your account",
    "confirm your password",
    "sign in"
  ];

  const foundWord = suspiciousWords.find(function (word) {
    return visibleText.includes(word);
  });

  if (foundWord) {
    score += 15;
    indicators.push(
      'The page contains the text "' + foundWord + '".'
    );
  }

  return {
    score: Math.min(score, 100),
    indicators: indicators
  };
}

chrome.runtime.onMessage.addListener(
  function (message, sender, sendResponse) {
    if (message.type === "ANALYZE_DOM") {
      sendResponse(analyzeDom());
    }
  }
);