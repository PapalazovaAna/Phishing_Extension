# Deployment

The backend runs on [Render](https://render.com) and the extension is
published in the Chrome Web Store. The backend has to be deployed first,
because the extension needs its address.

## 1. Backend on Render

The repository contains a [Render Blueprint](../render.yaml) that
describes the service, so almost nothing has to be set by hand.

1. Push the repository to GitHub.
2. In Render, choose **New > Blueprint** and select the repository.
   Render reads `render.yaml` and creates the web service
   `phishing-risk-detector`:
   - only the `backend` directory is deployed
   - the build installs `backend/requirements.txt` and downloads the
     current OpenPhish, Tranco and public suffix lists (if a download
     fails, the copies from the repository are used)
   - the server is started with `uvicorn` and checked with `/health`
3. When the deploy has finished, open
   `https://<service-name>.onrender.com/health`. It should return
   `{"status":"ok"}`. The interactive API documentation is at `/docs`.

If Render gives the service a different address than
`phishing-risk-detector.onrender.com` (the name may already be taken),
change the address in two places in the extension:
`BACKEND_URL` in `extension/config.js` and `host_permissions` in
`extension/manifest.json`.

**Free plan.** The service has 512 MB of memory; the backend uses about
260 MB. After about 15 minutes without requests the service is stopped,
and the next request starts it again, which takes up to a minute. The
extension waits up to 75 seconds and then shows a message that the
server is starting.

**Python version.** `render.yaml` sets `PYTHON_VERSION` to `3.13`. If
Render does not accept the short form, use a full version such as
`3.13.7`; the available versions are listed in Render's documentation.

**Allowed origins.** Chrome extensions can always call the API. Web
pages cannot, unless their origin is listed in the optional environment
variable `ALLOWED_ORIGINS` (comma-separated).

## 2. Extension in the Chrome Web Store

### Prepare the package

1. Check that `extension/config.js` and `extension/manifest.json`
   contain the Render address (see above).
2. Increase `version` in `manifest.json` for every new upload.
3. Test the extension once with the Render backend: load the `extension`
   folder in `chrome://extensions` (*Developer mode > Load unpacked*).
4. Create the ZIP file. The `manifest.json` must be at the top level of
   the ZIP, not inside a folder. In PowerShell, from the repository root:

   ```powershell
   Compress-Archive -Path extension\* -DestinationPath phishing-risk-detector.zip -Force
   ```

### Publish

1. Register in the
   [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole)
   (one-time registration fee).
2. **Add new item** and upload the ZIP file.
3. **Store listing:** description, category (for example *Tools*), at
   least one screenshot (1280 x 800 or 640 x 400) and a small promo tile
   (440 x 280). The icon is taken from the package.
4. **Privacy practices:** fill in the form with the answers below.
5. **Submit for review.** The review usually takes a few days.

### Answers for the privacy form

**Single purpose:**
> Warns users about phishing web pages by estimating a phishing risk
> score for the current page from its address, its content and
> reputation lists.

**Permission justifications:**

| Permission | Justification |
|---|---|
| `tabs` | Read the address of a page when it finishes loading, to check it for phishing before the user enters data. |
| `storage` | Remember for the current browser session that the user chose "Continue anyway" on a warning page. |
| Host permission for the backend | Send the page address to the extension's own analysis server, which calculates the risk score. |
| Content script on all `http`/`https` pages | Check the page for phishing features (password fields, forms that send data to another website) and show the warning page. |

**Remote code:** No, the extension does not use remote code.

**Data usage:** select *Web history* (the addresses of visited pages are
sent to the server) and *Website content* (short descriptions of page
features, such as "contains a password field", are sent when the popup is
opened). Confirm that the data is not sold, not used for unrelated
purposes and not used for creditworthiness.

**Privacy policy URL:** the extension handles browsing data, so a privacy
policy is required. Use [privacy-policy.md](privacy-policy.md): fill in
the two TODO lines (date and contact email) and link to the file on
GitHub, for example
`https://github.com/<user>/<repository>/blob/main/docs/privacy-policy.md`.
The repository must be public for the link to work.
