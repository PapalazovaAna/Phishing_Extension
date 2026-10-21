// Address of the backend API.
//
// For the published extension this is the Render service. For local
// development use "http://127.0.0.1:8000". The address must also be
// listed in "host_permissions" in manifest.json.
const BACKEND_URL = "https://phishing-risk-detector.onrender.com";

// A free Render server sleeps when it is not used and needs up to a
// minute to start again, so requests may take that long.
const BACKEND_TIMEOUT_MS = 75000;
