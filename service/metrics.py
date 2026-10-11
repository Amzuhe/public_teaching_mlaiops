"""Lab 4: Prometheus metrics for the dashboard, in the plain text format (no extra dependency).

Exposes the five required signals: request rate and error rate (counter by status class),
latency p50/p95/p99 (histogram), the model version in production (info gauge), and a
feature-distribution statistic over a rolling window (mean and PSI of the last N inputs).

shortcut: counters live in process memory, so with several uvicorn workers each worker
reports its own share. Fine for a single-worker local run; use prometheus_client's
multiprocess mode (or one worker per container) before scaling out.
"""
from __future__ import annotations

import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

# Histogram buckets in seconds, chosen around the 200 ms p95 target in monitoring/slo.yaml.
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5)
# Same window the drift detector is calibrated for (see README, "Drift threshold").
WINDOW = int(os.environ.get("DRIFT_WINDOW", "500"))

requests_total: dict[tuple[str, str], int] = defaultdict(int)
latency_counts: dict[str, list[int]] = defaultdict(lambda: [0] * len(BUCKETS))
latency_sum: dict[str, float] = defaultdict(float)
latency_n: dict[str, int] = defaultdict(int)
recent_inputs: deque[dict] = deque(maxlen=WINDOW)
_reference: dict[str, np.ndarray] | None = None


def observe(route: str, status: int, seconds: float) -> None:
    requests_total[(route, f"{status // 100}xx")] += 1
    for i, edge in enumerate(BUCKETS):
        if seconds <= edge:
            latency_counts[route][i] += 1
    latency_sum[route] += seconds
    latency_n[route] += 1


def record_inputs(rows: list[dict]) -> None:
    recent_inputs.extend(rows)


def _reference_columns(features: list[str]) -> dict[str, np.ndarray] | None:
    """The training window the rolling PSI compares against; loaded once, lazily."""
    global _reference
    if _reference is None:
        path = Path(os.environ.get("REFERENCE_PATH", "data/raw/sensors.csv"))
        if not path.exists():
            return None
        import pandas as pd

        frame = pd.read_csv(path)
        _reference = {f: frame[f].to_numpy(dtype=float) for f in features}
    return _reference


def render(model_version: str) -> str:
    from monitoring.drift import psi
    from src.data import FEATURES

    out = ["# TYPE http_requests_total counter"]
    for (route, status_class), n in sorted(requests_total.items()):
        out.append(f'http_requests_total{{route="{route}",status_class="{status_class}"}} {n}')

    out.append("# TYPE http_request_duration_seconds histogram")
    for route, counts in sorted(latency_counts.items()):
        for edge, n in zip(BUCKETS, counts):
            out.append(f'http_request_duration_seconds_bucket{{route="{route}",le="{edge}"}} {n}')
        out.append(f'http_request_duration_seconds_bucket{{route="{route}",le="+Inf"}} {latency_n[route]}')
        out.append(f'http_request_duration_seconds_sum{{route="{route}"}} {latency_sum[route]}')
        out.append(f'http_request_duration_seconds_count{{route="{route}"}} {latency_n[route]}')

    out.append("# TYPE model_version_info gauge")
    out.append(f'model_version_info{{version="{model_version}"}} 1')

    out.append("# TYPE feature_rolling_mean gauge")
    out.append("# TYPE feature_rolling_psi gauge")
    out.append(f"feature_window_size {len(recent_inputs)}")
    if recent_inputs:
        reference = _reference_columns(FEATURES)
        for f in FEATURES:
            current = np.array([row[f] for row in recent_inputs], dtype=float)
            out.append(f'feature_rolling_mean{{feature="{f}"}} {current.mean():.4f}')
            if reference is not None and len(current) >= 50:
                out.append(f'feature_rolling_psi{{feature="{f}"}} {psi(reference[f], current):.5f}')
    return "\n".join(out) + "\n"
