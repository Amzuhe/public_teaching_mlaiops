"""Lab 5 — register step. Only runs when the evaluation gate passed (pipeline condition).

Registers the model from the training run and points the 'production' alias at it, so the
next pipeline run's gate compares against this version.

    python scripts/register.py --metrics reports/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402

from src import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.metrics.read_text())
    cfg = config.load(strict=False)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    mv = mlflow.register_model(f"runs:/{payload['run_id']}/model", cfg.model_registry_name)
    client = mlflow.MlflowClient()
    # lineage: which data and which run this version came from
    client.set_model_version_tag(mv.name, mv.version, "data_fingerprint", payload["data_fingerprint"])
    client.set_registered_model_alias(mv.name, "production", mv.version)
    print(f"registered {mv.name} v{mv.version} (run {payload['run_id']}) -> alias production")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
