"""Lab 5 — run pipeline/pipeline.yaml locally, step by step, honouring its rules.

    python scripts/run_pipeline.py

Same DAG and same step commands the managed pipeline would run, executed on this machine.
  * a step with `on_failure: abort` that fails stops the whole run (exit 1)
  * `condition: "gate_decision == 'pass'"` runs the step only if the evaluate step exited 0
  * a skipped step skips everything that depends on it

shortcut: runs steps on the host, not in containers on managed compute; Azure ML pipelines
need a workspace that was not set up in time. Swap for cloudlayer.pipelines.compile_azure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudlayer.pipelines import load_spec  # noqa: E402

OUTPUTS = ROOT / "reports" / "pipeline_outputs"


def main() -> int:
    spec = load_spec()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    state: dict[str, str] = {}  # step name -> "ok" | "failed" | "skipped"
    gate = None

    for step in spec["steps"]:
        name = step["name"]
        deps = step.get("depends_on", [])
        if any(state.get(d) != "ok" for d in deps):
            state[name] = "skipped"
            print(f"[{name}] SKIPPED (upstream did not succeed)")
            continue
        cond = step.get("condition")
        if cond:
            if cond != "gate_decision == 'pass'":
                raise ValueError(f"unsupported condition {cond!r}")
            if gate != "pass":
                state[name] = "skipped"
                print(f"[{name}] SKIPPED (gate_decision = {gate})")
                continue

        # step commands write to /outputs inside a container; map it to a local folder
        cmd = [c.replace("/outputs", str(OUTPUTS)) for c in step["command"]]
        if cmd[0] == "python":
            cmd[0] = sys.executable
        elif cmd[0] == "pytest":
            cmd = [sys.executable, "-m", "pytest", *cmd[1:]]
        if name == "register":  # register needs the training run id from metrics.json
            cmd += ["--metrics", str(OUTPUTS / "metrics.json")]
        print(f"[{name}] $ {' '.join(cmd[1:] if cmd[0] == sys.executable else cmd)}")
        rc = subprocess.run(cmd, cwd=ROOT).returncode

        if name == "evaluate":
            gate = "pass" if rc == 0 else "fail"
            state[name] = "ok"  # a failing gate is a decision, not a crash
            print(f"[{name}] gate_decision = {gate}")
            continue
        state[name] = "ok" if rc == 0 else "failed"
        if rc != 0:
            print(f"[{name}] FAILED (exit {rc})")
            if step.get("on_failure") == "abort":
                print("PIPELINE ABORTED")
                return 1

    print("summary:", ", ".join(f"{k}={v}" for k, v in state.items()))
    return 0 if all(v != "failed" for v in state.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
