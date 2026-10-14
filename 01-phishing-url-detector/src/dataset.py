"""Reproducible dataset generation for the phishing URL detector.

To keep the project fully runnable offline (and avoid shipping scraped data of
uncertain licensing), we *synthesize* a labeled dataset from:

  * a seed list of well-known legitimate domains, expanded with realistic
    benign paths/subdomains, and
  * phishing-style URLs generated from documented attack patterns
    (brand impersonation, IP hosts, lookalike domains, urgency keywords,
    shorteners, deep random paths, etc.).

This is an *educational* dataset. The generator is deterministic given a seed,
so results are reproducible. For a production system you would swap this out
for a labeled feed (e.g. an authorized URL-reputation dataset).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# A small seed of popular, legitimate registered domains.
LEGIT_DOMAINS = (
    "google.com", "youtube.com", "wikipedia.org", "amazon.com", "github.com",
    "microsoft.com", "apple.com", "stackoverflow.com", "reddit.com",
    "linkedin.com", "netflix.com", "nytimes.com", "bbc.co.uk", "cloudflare.com",
    "python.org", "mozilla.org", "paypal.com", "dropbox.com", "spotify.com",
    "adobe.com", "wordpress.org", "yahoo.com", "ebay.com", "cnn.com",
    "harvard.edu", "mit.edu", "nasa.gov", "who.int", "imdb.com", "etsy.com",
)

LEGIT_PATHS = (
    "", "/", "/about", "/help", "/search?q=books", "/login", "/account/settings",
    "/products/1234", "/blog/2024/update", "/docs/getting-started",
    "/news/world", "/user/profile", "/support/contact", "/cart", "/watch?v=abc123",
    # Longer / noisier-but-benign paths so legit URLs aren't trivially short
    # and clean. Real product and tracking URLs look messy too.
    "/catalog/category/electronics/laptops?page=3&sort=price",
    "/items/9f8a7b6c/details?ref=home&utm_source=email",
    "/u/238471/posts/2024-10-01-release-notes",
    "/signin?redirect=%2Faccount%2Fsecurity&session=42",
)

LEGIT_SUBDOMAINS = ("", "www.", "mail.", "docs.", "support.", "blog.", "shop.")

# A few legitimate sites really do use newer TLDs. Including some keeps the
# classifier from treating a modern TLD as a perfect phishing signal.
LEGIT_MODERN = (
    "vercel.app", "openai.com", "notion.so", "figma.com", "replit.com",
    "example.io", "startup.xyz", "docs.page", "my.dev",
)

# Brands commonly impersonated, plus lookalike-domain building blocks.
BRANDS = (
    "paypal", "apple", "microsoft", "amazon", "netflix", "google", "facebook",
    "instagram", "bankofamerica", "wellsfargo", "chase", "dhl", "fedex",
    "coinbase", "binance", "outlook", "office365", "icloud",
)

PHISH_TLDS = ("xyz", "top", "click", "tk", "ml", "gq", "cf", "work", "zip", "link")
# Phishers also heavily abuse common TLDs (compromised sites, cheap domains),
# so a sizeable share of phishing URLs use these — removing the easy shortcut.
COMMON_TLDS = ("com", "net", "org", "info", "co", "online", "site")
PHISH_WORDS = (
    "login", "verify", "secure", "account", "update", "confirm", "signin",
    "billing", "unlock", "support", "webscr", "recovery", "alert",
)
SHORTENER_HOSTS = ("bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy")


def _rand_token(rng: random.Random, n_min: int = 6, n_max: int = 14) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    length = rng.randint(n_min, n_max)
    return "".join(rng.choice(alphabet) for _ in range(length))


def _phish_tld(rng: random.Random) -> str:
    """Pick a TLD for a phishing URL: ~55% abuse-heavy, ~45% common.

    This deliberate overlap with legitimate TLDs makes the classification
    problem non-trivial (no single feature separates the classes)."""
    if rng.random() < 0.55:
        return rng.choice(PHISH_TLDS)
    return rng.choice(COMMON_TLDS)


def _make_legit(rng: random.Random) -> str:
    # ~15% of legitimate URLs use a modern TLD on a real service.
    if rng.random() < 0.15:
        sub = rng.choice(("", "www.", "app.", "docs."))
        domain = rng.choice(LEGIT_MODERN)
    else:
        sub = rng.choice(LEGIT_SUBDOMAINS)
        domain = rng.choice(LEGIT_DOMAINS)
    path = rng.choice(LEGIT_PATHS)
    scheme = "https" if rng.random() < 0.9 else "http"
    return f"{scheme}://{sub}{domain}{path}"


def _make_phish(rng: random.Random) -> str:
    """Generate a phishing-style URL using one of several attack patterns."""
    pattern = rng.choice(
        ["ip_host", "lookalike", "brand_subdomain", "shortener",
         "random_host", "at_trick", "deep_path"]
    )
    brand = rng.choice(BRANDS)
    word = rng.choice(PHISH_WORDS)
    # Phishing increasingly uses HTTPS (free certs), so don't let the scheme
    # be a giveaway — roughly half are https.
    scheme = "https" if rng.random() < 0.5 else "http"

    if pattern == "ip_host":
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        port = f":{rng.choice([8080, 8000, 8443, 81])}" if rng.random() < 0.5 else ""
        return f"{scheme}://{ip}{port}/{brand}/{word}.php?id={_rand_token(rng)}"

    if pattern == "lookalike":
        tweak = rng.choice([
            brand.replace("o", "0"), brand.replace("l", "1"),
            brand + "-" + word, brand + word, "secure-" + brand,
            brand.replace("a", "4"),
        ])
        tld = _phish_tld(rng)
        return f"{scheme}://{tweak}.{tld}/{word}"

    if pattern == "brand_subdomain":
        tld = _phish_tld(rng)
        host = _rand_token(rng, 5, 9)
        return f"{scheme}://{brand}.{word}.{host}.{tld}/{word}/signin"

    if pattern == "shortener":
        host = rng.choice(SHORTENER_HOSTS)
        return f"{scheme}://{host}/{_rand_token(rng, 5, 8)}"

    if pattern == "random_host":
        tld = _phish_tld(rng)
        return f"{scheme}://{_rand_token(rng, 10, 20)}.{tld}/{word}/{brand}"

    if pattern == "at_trick":
        # The '@' trick: everything before @ is ignored by the browser.
        tld = _phish_tld(rng)
        return (f"{scheme}://{brand}.com@{_rand_token(rng, 6, 10)}.{tld}"
                f"/{word}")

    # deep_path
    tld = _phish_tld(rng)
    depth = "/".join(_rand_token(rng, 4, 8) for _ in range(rng.randint(3, 5)))
    return f"{scheme}://{_rand_token(rng, 8, 14)}.{tld}/{depth}/{brand}-{word}"


def generate_dataset(
    n_per_class: int = 2000, seed: int = 42
) -> list[tuple[str, int]]:
    """Return a shuffled list of (url, label) where label 1 = phishing."""
    rng = random.Random(seed)
    rows: list[tuple[str, int]] = []
    seen: set[str] = set()

    def add(url: str, label: int) -> None:
        if url not in seen:
            seen.add(url)
            rows.append((url, label))

    # Over-generate to compensate for de-duplication, then trim per class.
    legit, phish = [], []
    attempts = 0
    while (len(legit) < n_per_class or len(phish) < n_per_class) and attempts < n_per_class * 50:
        attempts += 1
        if len(legit) < n_per_class:
            u = _make_legit(rng)
            if u not in seen:
                seen.add(u)
                legit.append((u, 0))
        if len(phish) < n_per_class:
            u = _make_phish(rng)
            if u not in seen:
                seen.add(u)
                phish.append((u, 1))

    rows = legit[:n_per_class] + phish[:n_per_class]
    rng.shuffle(rows)
    return rows


def write_csv(path: str | Path, rows: list[tuple[str, int]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate the phishing URL dataset.")
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/urls.csv")
    args = ap.parse_args()

    data = generate_dataset(args.n_per_class, args.seed)
    out = write_csv(args.out, data)
    print(f"Wrote {len(data)} rows ({args.n_per_class} per class) to {out}")
