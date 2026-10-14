"""Load and preprocess the NSL-KDD dataset.

Responsibilities:
  * attach the 41 feature names + label + difficulty columns,
  * map the ~40 fine-grained attack labels to the 4 standard attack
    categories (DoS, Probe, R2L, U2R) plus ``normal``,
  * expose both a binary target (normal vs attack) and a multi-class target.

Categorical features (protocol_type, service, flag) are returned as-is;
encoding is handled in the training pipeline so train/test share one encoder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# The 41 NSL-KDD features in order, then the label and the difficulty score.
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]

# Standard mapping of fine-grained attack names to the 4 NSL-KDD categories.
ATTACK_CATEGORIES = {
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L",
    "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}


def _categorize(label: str) -> str:
    if label == "normal":
        return "normal"
    return ATTACK_CATEGORIES.get(label, "unknown")


def load_split(name: str) -> pd.DataFrame:
    """Load a single NSL-KDD file into a labeled DataFrame.

    Adds columns: ``category`` (normal/DoS/Probe/R2L/U2R/unknown) and
    ``is_attack`` (0 = normal, 1 = attack).
    """
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.download_data"
        )
    df = pd.read_csv(path, names=COLUMNS)
    df["category"] = df["label"].map(_categorize)
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_split("KDDTrain+.txt"), load_split("KDDTest+.txt")


def feature_columns() -> list[str]:
    """The 41 model input columns (everything but label/difficulty/derived)."""
    drop = {"label", "difficulty", "category", "is_attack"}
    return [c for c in COLUMNS if c not in drop]
