"""CICIDS2017 (CICFlowMeter) schema constants and label handling."""

from __future__ import annotations

import re

LABEL_RAW = "label_raw"
BENIGN = "Benign"

# The 78 CICFlowMeter flow features present in the CICIDS2017 CSVs (after stripping header whitespace).
CICIDS_FEATURES: list[str] = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min",
]

# CICFlowMeter writes this column twice (identical values), pandas renames the second to ".1".
DUPLICATE_COLUMNS = ["Fwd Header Length.1"]

# Canonical ordering of the grouped ("family") classes. Benign is always class index 0.
FAMILY_ORDER = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration", "Heartbleed"]

# normalised raw label -> (family, pretty raw name)
_LABELS: dict[str, tuple[str, str]] = {
    "benign": ("Benign", "Benign"),
    "dos hulk": ("DoS", "DoS Hulk"),
    "dos goldeneye": ("DoS", "DoS GoldenEye"),
    "dos slowloris": ("DoS", "DoS slowloris"),
    "dos slowhttptest": ("DoS", "DoS Slowhttptest"),
    "ddos": ("DDoS", "DDoS"),
    "portscan": ("PortScan", "PortScan"),
    "ftp patator": ("BruteForce", "FTP-Patator"),
    "ssh patator": ("BruteForce", "SSH-Patator"),
    "web attack brute force": ("WebAttack", "WebAttack-BruteForce"),
    "web attack xss": ("WebAttack", "WebAttack-XSS"),
    "web attack sql injection": ("WebAttack", "WebAttack-SQLi"),
    "bot": ("Bot", "Bot"),
    "infiltration": ("Infiltration", "Infiltration"),
    "heartbleed": ("Heartbleed", "Heartbleed"),
}


def normalise_label(raw: object) -> str:
    """Lower-case and collapse punctuation. Robust to the mangled en-dash in 'Web Attack � XSS'."""
    return re.sub(r"[^a-z0-9]+", " ", str(raw).lower()).strip()


def map_label(raw: object, grouping: str = "family") -> str:
    """Map a raw CICIDS2017 label to a class name under the chosen grouping."""
    norm = normalise_label(raw)
    if norm in _LABELS:
        family, pretty = _LABELS[norm]
        return family if grouping == "family" else pretty
    return norm.title().replace(" ", "")  # unknown labels survive as their own class


def order_classes(names: list[str]) -> list[str]:
    """Benign first, then canonical family order, then anything else alphabetically."""
    known = [n for n in FAMILY_ORDER if n in names]
    rest = sorted(n for n in names if n not in FAMILY_ORDER)
    # raw grouping: put Benign first, keep the rest alphabetical
    if "Benign" in rest:
        rest.remove("Benign")
        known = ["Benign"] + known
    return known + rest
