"""Lab 5 — the registration gate.

Compares a candidate against the currently registered production model. Exits non-zero
when the candidate should NOT be registered, which is what makes the pipeline condition
real rather than decorative.

    python scripts/evaluation_gate.py --metrics reports/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Measured 2026-10-23: test ROC-AUC over 6 seeds = 0.847 mean, 0.017 SD (reports/noise/).
# MIN_ABSOLUTE ~ mean - 3 SD: below it the model is broken, not unlucky.
# MIN_IMPROVEMENT ~ 0.6 SD: the old 0.002 was ~1/8 of the seed noise, so luck alone would
# pass it. 0.01 still lets a real gain ship without waiting for a full 1-SD jump.
MIN_ABSOLUTE = 0.80
MIN_IMPROVEMENT = 0.01
ALIAS = "production"


def registry_incumbent(metric: str) -> float | None:
    """The metric of the model currently aliased 'production', or None on a first run."""
    import mlflow
    from mlflow.exceptions import MlflowException

    from src import config

    cfg = config.load(strict=False)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(cfg.model_registry_name, ALIAS)
    except MlflowException:
        return None  # no registered model or no production alias yet: legitimately none
    return float(client.get_run(mv.run_id).data.metrics[metric])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--metric", default="test_roc_auc")
    ap.add_argument("--incumbent", type=float, default=None,
                    help="current production score; omit to read from the registry")
    args = ap.parse_args()

    payload = json.loads(args.metrics.read_text())
    candidate = float(payload[args.metric])
    print(f"candidate {args.metric} = {candidate:.4f}")

    if candidate < MIN_ABSOLUTE:
        print(f"GATE FAIL  below absolute floor {MIN_ABSOLUTE}")
        return 1

    incumbent = args.incumbent
    if incumbent is None:
        incumbent = registry_incumbent(args.metric)
    if incumbent is None:
        print("no production model in the registry — first registration")
        print("GATE PASS")
        return 0

    delta = candidate - incumbent
    print(f"incumbent {incumbent:.4f}  delta {delta:+.4f}  required {MIN_IMPROVEMENT:+.4f}")
    if delta < MIN_IMPROVEMENT:
        print("GATE FAIL  improvement within noise; not registering")
        return 1

    print("GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
