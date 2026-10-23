"""Lab 5: minimal load test (stdlib only) when k6/locust are not installed.

    python loadtest/simple_load.py --target http://127.0.0.1:8080 --users 10 --seconds 30

Prints requests/second, error count and p50/p95 latency for single /predict calls.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def payload() -> bytes:
    return json.dumps({
        "temp_c": round(random.uniform(60, 95), 3),
        "vibration_mm_s": round(random.uniform(1.0, 8.0), 3),
        "pressure_kpa": round(random.uniform(280, 350), 3),
        "hours_since_service": round(random.uniform(0, 9000), 3),
        "load_pct": round(random.uniform(20, 100), 3),
        "ambient_humidity": round(random.uniform(30, 85), 3),
    }).encode()


def worker(url: str, until: float) -> tuple[list[float], int]:
    lat, errors = [], 0
    while time.perf_counter() < until:
        req = urllib.request.Request(url, data=payload(), headers={"Content-Type": "application/json"})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            lat.append(time.perf_counter() - t)
        except Exception:
            errors += 1
    return lat, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://127.0.0.1:8080")
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=30)
    a = ap.parse_args()
    until = time.perf_counter() + a.seconds
    with ThreadPoolExecutor(a.users) as ex:
        results = list(ex.map(lambda _: worker(a.target + "/predict", until), range(a.users)))
    lat = sorted(x for r in results for x in r[0])
    errors = sum(r[1] for r in results)
    assert lat, "no successful requests"
    print(f"users={a.users} requests={len(lat)} errors={errors} rps={len(lat) / a.seconds:.1f} "
          f"p50={lat[len(lat) // 2] * 1000:.0f}ms p95={lat[int(len(lat) * 0.95)] * 1000:.0f}ms")


if __name__ == "__main__":
    main()
