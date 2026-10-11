"""Lab 4: send realistic traffic to the running service so the dashboard has something to show.

    python scripts/traffic.py --n 600                 # healthy traffic, ~2% bad requests (4xx)
    python scripts/traffic.py --n 600 --shift 6       # temp_c shifted +6 C: watch the PSI panel

Rows are sampled across the whole fleet from the training data (stdlib HTTP, no extra deps).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import data  # noqa: E402


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/predict")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--shift", type=float, default=0.0, help="add this many degrees to temp_c")
    ap.add_argument("--bad-rate", type=float, default=0.02, help="share of malformed requests (4xx)")
    ap.add_argument("--rps", type=float, default=20.0)
    args = ap.parse_args()

    rows = pd.read_csv("data/raw/sensors.csv")[data.FEATURES].sample(n=args.n, replace=True, random_state=None)
    codes: dict[int, int] = {}
    for row in rows.to_dict("records"):
        row["temp_c"] = row["temp_c"] + args.shift
        body = {**row, "surprise": 1} if random.random() < args.bad_rate else row
        code = post(args.url, body)
        codes[code] = codes.get(code, 0) + 1
        time.sleep(1 / args.rps)
    print(f"sent {args.n} requests (temp_c shift {args.shift:+}): {codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
