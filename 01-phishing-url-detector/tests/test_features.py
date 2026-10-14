"""Unit tests for lexical URL feature extraction."""

from src.features import (
    FEATURE_NAMES,
    extract_features,
    features_to_vector,
    _shannon_entropy,
)


def test_vector_length_matches_names():
    vec = features_to_vector("https://example.com")
    assert len(vec) == len(FEATURE_NAMES)


def test_feature_dict_has_all_names_in_order():
    feats = extract_features("https://example.com/path")
    assert list(feats.keys()) == list(FEATURE_NAMES)


def test_https_detected():
    assert extract_features("https://example.com")["uses_https"] == 1.0
    assert extract_features("http://example.com")["uses_https"] == 0.0


def test_ip_host_detected():
    f = extract_features("http://192.168.0.1/login")
    assert f["has_ip_host"] == 1.0
    assert extract_features("http://example.com")["has_ip_host"] == 0.0


def test_at_symbol_counted():
    f = extract_features("http://paypal.com@evil.xyz/verify")
    assert f["num_at"] == 1.0


def test_shortener_detected():
    assert extract_features("http://bit.ly/abc123")["is_shortener"] == 1.0
    assert extract_features("http://github.com")["is_shortener"] == 0.0


def test_suspicious_keywords_counted():
    f = extract_features("http://secure-login-verify.xyz/account")
    assert f["num_suspicious_keywords"] >= 3


def test_suspicious_tld():
    assert extract_features("http://foo.xyz")["suspicious_tld"] == 1.0
    assert extract_features("http://foo.com")["suspicious_tld"] == 0.0


def test_subdomain_count():
    assert extract_features("http://a.b.c.example.com")["num_subdomains"] == 3.0
    assert extract_features("http://example.com")["num_subdomains"] == 0.0


def test_punycode_detected():
    assert extract_features("http://xn--pple-43d.com")["has_punycode"] == 1.0


def test_entropy_monotonic():
    low = _shannon_entropy("aaaaaaaa")
    high = _shannon_entropy("a1b2c3d4")
    assert high > low
    assert _shannon_entropy("") == 0.0


def test_no_scheme_defaults_gracefully():
    # Should not raise and should still parse a hostname.
    f = extract_features("example.com/login")
    assert f["hostname_length"] > 0


def test_digit_ratio_bounds():
    f = extract_features("http://12345.xyz")
    assert 0.0 <= f["digit_ratio"] <= 1.0
