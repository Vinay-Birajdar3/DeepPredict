"""Generates a valid sample sequence payload for testing the prediction API."""

import json
from pathlib import Path
import pandas as pd
from src.config import ALL_RAW_COLUMNS, REPORTS_DIR

def generate_sample_payload():
    test_file = Path("data/raw/test_FD001.txt")
    if not test_file.exists():
        print("test_FD001.txt not found")
        return

    df = pd.read_csv(test_file, sep=r"\s+", header=None).iloc[:, :len(ALL_RAW_COLUMNS)]
    df.columns = ALL_RAW_COLUMNS

    # Extract last 30 cycles of Engine 1 across the 14 dynamic features
    features = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    e1_seq = df[df["unit_number"] == 1][features].tail(30).to_numpy().tolist()

    out_path = REPORTS_DIR / "sample_sequence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(e1_seq, f, indent=2)

    print(f"Sample sequence saved to {out_path} (Shape: {len(e1_seq)}x{len(e1_seq[0])})")

if __name__ == "__main__":
    generate_sample_payload()

