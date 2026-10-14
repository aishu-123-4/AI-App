"""Download the NSL-KDD dataset.

NSL-KDD is the de-facto benchmark for network intrusion detection — a cleaned
version of the classic KDD Cup 99 set with redundant records removed and a
harder, distribution-shifted test split. We fetch the official train/test
files from a public mirror into ``data/``.

Run:
    python -m src.download_data
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Public mirrors of the canonical NSL-KDD text files.
FILES = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
}


def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = DATA / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  exists  {name} ({dest.stat().st_size:,} bytes)")
            continue
        print(f"  fetch   {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved   {name} ({dest.stat().st_size:,} bytes)")
    print("Done.")


if __name__ == "__main__":
    download()
