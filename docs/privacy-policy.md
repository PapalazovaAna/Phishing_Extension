# Privacy Policy: Phishing Risk Detector

*Last updated: TODO date of publishing*

Phishing Risk Detector is a browser extension that estimates how likely a
web page is to be a phishing page. This policy explains which data the
extension processes and why.

## Data that is processed

**Addresses of visited pages.** When a page with an `http` or `https`
address finishes loading, the extension sends its full address (URL) to
the Phishing Risk Detector server to check whether it is a phishing page.
This is the only way the automatic warning can work. Pages on `localhost`
and browser-internal pages (for example `chrome://`) are not sent.

**Analysis results of the page.** When you open the extension popup, the
extension examines the open page in your browser and sends the results
together with the URL: a number from 0 to 100 and short descriptions
such as "The page contains a password input field".

**Not collected:** the content of the page, anything you type into forms,
passwords, cookies, your browsing history, your name, email address or
any other information that identifies you.

## How the data is used

The server uses the URL and the analysis results only to calculate the
risk score and sends it back to the extension. The URL is processed in
memory and is not stored, logged or used for anything else. The server
does not use cookies or identify users.

The server is hosted on [Render](https://render.com). Like any web
service, Render may record technical connection data (such as IP
addresses and request times) in its own logs; these logs do not contain
the URLs of the pages you visit.

## Data stored in the browser

When you choose "Continue anyway" on a warning page, the extension
remembers this for the current browser session, so the warning is not
shown again for that page. This information stays in your browser and is
deleted when the browser is closed.

## Sharing

The data is not sold, shared with third parties, or used for advertising,
profiling or any purpose other than the phishing check described above.

## Changes

If this policy changes, the new version will be published at the same
address with a new date.

## Contact

Questions about this policy: TODO contact email

This extension was developed as part of a diploma thesis at the Faculty
of Computer Science and Engineering (FINKI), Ss. Cyril and Methodius
University in Skopje.
