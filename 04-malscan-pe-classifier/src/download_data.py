"""Download the ClaMP PE-header malware dataset.

ClaMP (Classification of Malware with PE headers) provides static features
extracted from the PE headers of ~5,200 Windows executables, labeled benign
(0) or malware (1). Static features mean **no code is ever executed** — safe to
work with.

Run:
    python -m src.download_data
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
URL = ("https://raw.githubusercontent.com/urwithajit9/ClaMP/master/"
       "dataset/ClaMP_Integrated-5184.csv")
DEST = DATA / "ClaMP_Integrated-5184.csv"


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
