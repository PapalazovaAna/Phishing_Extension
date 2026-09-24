import csv
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


DATA_DIR = Path(__file__).parent / "reputation_data"

OPENPHISH_URL = (
    "https://raw.githubusercontent.com/openphish/public_feed/main/feed.txt"
)

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


def download_file(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "PhishingRiskDetector/1.0"
        },
    )

    with urlopen(request, timeout=60) as response:
        return response.read()


def download_openphish() -> None:
    print("Downloading OpenPhish feed...")

    content = download_file(OPENPHISH_URL).decode(
        "utf-8",
        errors="ignore",
    )

    phishing_urls = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
        and not line.startswith("#")
    ]

    output_path = DATA_DIR / "phishing_urls.txt"

    output_path.write_text(
        "\n".join(phishing_urls),
        encoding="utf-8",
    )

    print(
        f"Saved {len(phishing_urls)} phishing URLs to "
        f"{output_path}"
    )


def download_tranco() -> None:
    print("Downloading Tranco list...")

    archive_content = download_file(TRANCO_URL)

    with zipfile.ZipFile(
        io.BytesIO(archive_content)
    ) as archive:
        csv_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".csv")
        )

        with archive.open(csv_name) as csv_file:
            text_file = io.TextIOWrapper(
                csv_file,
                encoding="utf-8",
            )

            reader = csv.reader(text_file)

            domains = []

            for row in reader:
                if len(row) < 2:
                    continue

                domain = row[1].strip().lower()

                if domain:
                    domains.append(domain)

    output_path = DATA_DIR / "popular_domains.txt"

    output_path.write_text(
        "\n".join(domains),
        encoding="utf-8",
    )

    print(
        f"Saved {len(domains)} popular domains to "
        f"{output_path}"
    )


if __name__ == "__main__":
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    download_openphish()
    download_tranco()

    print("Reputation data download completed.")