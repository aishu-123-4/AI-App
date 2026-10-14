"""Tests for the sshd log parser and per-IP feature aggregation."""

from src.features import events_to_ip_table
from src.parse import parse_line, parse_lines

SAMPLE = [
    "Dec 10 06:55:46 LabSZ sshd[24200]: Failed password for root from 1.2.3.4 port 50 ssh2",
    "Dec 10 06:55:48 LabSZ sshd[24201]: Failed password for invalid user admin from 1.2.3.4 port 51 ssh2",
    "Dec 10 06:55:49 LabSZ sshd[24202]: Invalid user admin from 1.2.3.4",
    "Dec 10 06:56:00 LabSZ sshd[24300]: Accepted password for alice from 9.9.9.9 port 22 ssh2",
    "Dec 10 06:56:01 LabSZ sshd[24301]: reverse mapping checking getaddrinfo for x [5.5.5.5] failed - POSSIBLE BREAK-IN ATTEMPT!",
    "Dec 10 06:56:02 LabSZ sshd[24302]: Some unrelated message",
]


def test_parse_failed_password():
    ev = parse_line(SAMPLE[0])
    assert ev is not None
    assert ev.event_type == "failed_password"
    assert ev.ip == "1.2.3.4"
    assert ev.user == "root"
    assert ev.pid == 24200


def test_parse_invalid_user():
    ev = parse_line(SAMPLE[1])
    assert ev.event_type == "failed_invalid_user"
    assert ev.user == "admin"


def test_parse_accepted():
    ev = parse_line(SAMPLE[3])
    assert ev.event_type == "accepted"
    assert ev.ip == "9.9.9.9"
    assert ev.user == "alice"


def test_parse_break_in():
    ev = parse_line(SAMPLE[4])
    assert ev.event_type == "break_in"
    assert ev.ip == "5.5.5.5"


def test_unrecognized_message_is_other():
    ev = parse_line(SAMPLE[5])
    assert ev.event_type == "other"
    assert ev.ip is None


def test_non_sshd_line_returns_none():
    assert parse_line("Dec 10 06:55:46 LabSZ kernel: boot") is None


def test_timestamp_parsed():
    ev = parse_line(SAMPLE[0])
    assert ev.ts is not None
    assert ev.ts.month == 12 and ev.ts.day == 10


def test_ip_table_aggregates():
    events = list(parse_lines(SAMPLE))
    df = events_to_ip_table(events)
    assert "1.2.3.4" in df.index
    attacker = df.loc["1.2.3.4"]
    assert attacker["total_events"] == 3
    assert attacker["failed_password"] == 1
    assert attacker["accepted"] == 0
    assert df.loc["9.9.9.9"]["accepted"] == 1
    # fail_ratio is between 0 and 1.
    assert 0.0 <= attacker["fail_ratio"] <= 1.0
