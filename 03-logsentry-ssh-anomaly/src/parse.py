"""Parse raw sshd log lines into structured events.

The parser recognizes the common OpenSSH auth messages and extracts, where
present, the source IP, the attempted username, and a normalized event type.
Unrecognized lines are returned with ``event_type="other"`` so nothing is
silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

IP = r"(?P<ip>\d{1,3}(?:\.\d{1,3}){3})"

# Ordered (event_type, compiled regex). First match wins.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("accepted", re.compile(rf"Accepted password for (?P<user>\S+) from {IP}")),
    ("accepted", re.compile(rf"Accepted publickey for (?P<user>\S+) from {IP}")),
    ("failed_invalid_user",
     re.compile(rf"Failed password for invalid user (?P<user>\S+) from {IP}")),
    ("failed_password",
     re.compile(rf"Failed password for (?P<user>\S+) from {IP}")),
    ("invalid_user", re.compile(rf"Invalid user (?P<user>\S+) from {IP}")),
    ("auth_failure",
     re.compile(rf"authentication failure;.*rhost={IP}(?:\s+user=(?P<user>\S+))?")),
    ("break_in",
     re.compile(rf"reverse mapping checking getaddrinfo .*\[{IP}\] failed "
                r"- POSSIBLE BREAK-IN ATTEMPT")),
    ("disconnect", re.compile(rf"Received disconnect from {IP}")),
    ("connection_closed", re.compile(rf"Connection closed by {IP}")),
    ("no_ident", re.compile(rf"Did not receive identification string from {IP}")),
]

# e.g. "Dec 10 06:55:46 LabSZ sshd[24200]: <message>"
LINE_RE = re.compile(
    r"^(?P<ts>[A-Za-z]{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+sshd\[(?P<pid>\d+)\]:\s+(?P<msg>.*)$"
)


@dataclass
class Event:
    ts: datetime | None
    host: str
    pid: int
    event_type: str
    ip: str | None
    user: str | None
    raw: str


def _parse_ts(ts: str, year: int = 2024) -> datetime | None:
    try:
        return datetime.strptime(f"{year} {ts}", "%Y %b %d %H:%M:%S")
    except ValueError:
        return None


def parse_line(line: str) -> Event | None:
    line = line.rstrip("\n")
    m = LINE_RE.match(line)
    if not m:
        return None
    msg = m.group("msg")
    event_type, ip, user = "other", None, None
    for etype, pat in PATTERNS:
        pm = pat.search(msg)
        if pm:
            event_type = etype
            gd = pm.groupdict()
            ip = gd.get("ip")
            user = gd.get("user")
            break
    return Event(
        ts=_parse_ts(m.group("ts")),
        host=m.group("host"),
        pid=int(m.group("pid")),
        event_type=event_type,
        ip=ip,
        user=user,
        raw=line,
    )


def parse_lines(lines: Iterable[str]) -> Iterator[Event]:
    for line in lines:
        ev = parse_line(line)
        if ev is not None:
            yield ev


def parse_file(path: str | Path) -> list[Event]:
    path = Path(path)
    with path.open(encoding="utf-8", errors="replace") as fh:
        return list(parse_lines(fh))
