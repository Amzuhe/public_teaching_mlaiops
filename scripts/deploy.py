"""Lab 5 — deploy step.

shortcut: only points the 'staging' alias at the production version; a real Container Apps
deploy needs the Lab 3 serving image, which is not built yet. Replace once Lab 3 is done.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402

from src import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="staging")
    args = ap.parse_args()

    cfg = config.load(strict=False)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(cfg.model_registry_name, "production")
    client.set_registered_model_alias(cfg.model_registry_name, args.target, mv.version)
    print(f"{cfg.model_registry_name} v{mv.version} -> alias {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
