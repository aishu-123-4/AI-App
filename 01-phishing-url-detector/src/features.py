"""Lexical feature extraction for URLs.

Phishing URLs can often be distinguished from legitimate ones using only the
*text* of the URL — no network request required. This module turns a raw URL
string into a fixed-length numeric feature vector based on well-documented
lexical heuristics from phishing-detection research.

The features are intentionally interpretable so the trained model's decisions
can be explained (useful for a security context and for a model card).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

# Tokens frequently abused in phishing URLs to impersonate trusted services
# or create a false sense of urgency.
SUSPICIOUS_KEYWORDS = (
    "login", "signin", "verify", "verification", "account", "update",
    "secure", "security", "bank", "confirm", "password", "credential",
    "webscr", "ebayisapi", "payment", "invoice", "wallet", "unlock",
    "suspend", "recover", "billing", "support",
)

# URL shorteners hide the true destination — a common phishing tactic.
SHORTENERS = (
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
)

# TLDs disproportionately represented in abuse feeds (illustrative set).
SUSPICIOUS_TLDS = (
    "zip", "mov", "xyz", "top", "gq", "ml", "cf", "ga", "tk", "work",
    "click", "link", "country", "kim", "science", "party", "review",
)

IP_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)

# Ordered feature names — the training/serving code relies on this order.
FEATURE_NAMES = (
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_question",
    "num_ampersand",
    "num_equals",
    "num_digits",
    "num_subdomains",
    "digit_ratio",
    "has_ip_host",
    "uses_https",
    "has_port",
    "is_shortener",
    "suspicious_tld",
    "num_suspicious_keywords",
    "hostname_entropy",
    "has_punycode",
)


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of a string in bits/char. Random-looking hostnames
    (often auto-generated for phishing) have higher entropy."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _normalize(url: str) -> str:
    url = (url or "").strip()
    # urlparse needs a scheme to populate netloc reliably.
    if "://" not in url:
        url = "http://" + url
    return url


def extract_features(url: str) -> dict[str, float]:
    """Return the ordered lexical feature dict for a single URL."""
    raw = _normalize(url)
    parsed = urlparse(raw)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # Strip a trailing dot and split into labels for subdomain counting.
    labels = [p for p in hostname.split(".") if p]
    tld = labels[-1].lower() if labels else ""
    # subdomains = labels beyond the registered domain (domain + tld).
    num_subdomains = max(0, len(labels) - 2)

    digits = sum(ch.isdigit() for ch in raw)

    feats = {
        "url_length": float(len(raw)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path)),
        "num_dots": float(raw.count(".")),
        "num_hyphens": float(raw.count("-")),
        "num_at": float(raw.count("@")),
        "num_question": float(raw.count("?")),
        "num_ampersand": float(raw.count("&")),
        "num_equals": float(raw.count("=")),
        "num_digits": float(digits),
        "num_subdomains": float(num_subdomains),
        "digit_ratio": digits / len(raw) if raw else 0.0,
        "has_ip_host": 1.0 if IP_RE.match(hostname) else 0.0,
        "uses_https": 1.0 if parsed.scheme == "https" else 0.0,
        "has_port": 1.0 if parsed.port else 0.0,
        "is_shortener": 1.0 if hostname.lower() in SHORTENERS else 0.0,
        "suspicious_tld": 1.0 if tld in SUSPICIOUS_TLDS else 0.0,
        "num_suspicious_keywords": float(
            sum(kw in raw.lower() for kw in SUSPICIOUS_KEYWORDS)
        ),
        "hostname_entropy": _shannon_entropy(hostname),
        "has_punycode": 1.0 if "xn--" in hostname.lower() else 0.0,
    }
    # Guarantee declared order and completeness.
    return {name: feats[name] for name in FEATURE_NAMES}


def features_to_vector(url: str) -> list[float]:
    """Return features as a plain ordered list (model input)."""
    f = extract_features(url)
    return [f[name] for name in FEATURE_NAMES]
