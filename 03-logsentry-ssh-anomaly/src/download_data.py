"""Download a real SSH auth-log sample (Loghub / LabSZ OpenSSH logs).

These are genuine `sshd` logs captured on a server under sustained brute-force
attack — ideal for anomaly / brute-force detection.

Run:
    python -m src.download_data
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
URL = "https://raw.githubusercontent.com/logpai/loghub/master/OpenSSH/OpenSSH_2k.log"
DEST = DATA / "OpenSSH.log"


def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"  exists  {DEST.name} ({DEST.stat().st_size:,} bytes)")
        return
    print(f"  fetch   {DEST.name} ...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"  saved   {DEST.name} ({DEST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    download()
