# ITCS355 Lab 1 — Reproducible Training

> **Course materials live in [`course/`](course/README.md)** — syllabus, slides, the faculty
> specification, all five lab handouts, and the project brief. Every document is Markdown and
> renders on GitHub, diagrams included. New to the repo? Start with the
> [portability reference](course/reference/cloud-portability-reference.md).
> Keep this block when you edit the rest of this file; it is not part of the Lab 1 deliverable.

Predicting machine failure within 7 days from sensor readings. The model is not the point;
whether a stranger can reproduce it is.

> **This README is graded.** A grader with Docker and nothing else from your setup runs one
> command and compares the result against the claim below. Edit every `<...>` and delete the
> instruction blocks marked **REPLACE** before submitting.

---

## Reproduce

```bash
make reproduce
```

expected test_roc_auc: 0.848 ± 0.001

Runtime: about 40 seconds on 4 cores. No cloud account or credentials needed for this command —
that is deliberate, and it is why a grader can run it.

**REPLACE:** re-measure and update that claim line after your final change. Keep the exact
format `expected test_roc_auc: <value> ± <tolerance>`; `make verify` parses it, and so does the
grading script.

The tolerance covers the grader's machine, not your choice of seed. `make reproduce` pins the
seed for the dataset and the model both, so the grader runs your exact configuration; anything
they see that differs came from the environment — a different CPU, a different BLAS, a library
that moved under you. Measure it by running `make reproduce` twice, on two machines if you can
get them. On this pipeline it is tiny but not zero: the same seed gives 0.8482511 on an arm64
laptop and 0.8482379 on the amd64 CI runner — different architecture, different Python, same
pins — a gap of 1.3e-5. So ±0.001 is roughly seventy times more room than you need, and ±0.01
is padding.

Changing the seed measures something else. It moves the *split* — which machines land in test —
and that swings `test_roc_auc` across roughly 0.82–0.87 on this data. Write that number down in
your run discussion, because it says how much of your headline metric is luck. It is not your
tolerance.

---

## The problem

240 machines, 25 readings each, 6 sensor features, binary target `failed_within_7d` with a
positive rate near 12%.

Machines have persistent characteristics — a hot-running machine reads hot in every row. So the
train/validation/test split is **grouped by `machine_id`**: every reading from one machine lands
in exactly one partition. Splitting row-wise instead lets the model memorise the machine and
reports a validation score that will never survive production. `tests/test_data.py` asserts this
property holds, and Lab 4 turns it into a CI gate.

Bringing your own dataset is allowed. Replace `scripts/make_dataset.py`, update the schema in
`src/data.py`, and keep every test passing.

---

## Layout

```
src/          Layer 1 — provider-neutral. No SDKs, no bucket names, no absolute paths.
cloudlayer/   Layer 3 — the only place a provider SDK may be imported.
scripts/      Dataset generation, cloud check, portability audit, metric verification.
tests/        Data contract tests and split property tests.
```

`src/config.py` is the single point of environment knowledge. Everything else reads from it.
`make portability-audit` enforces the rule; it fails the build if a provider string appears in
`src/` or `tests/`.

---

## Setup

```bash
cp cloud.env.example cloud.env      # fill in, never commit
make setup
make cloud-check                    # eight slots, all PASS
make data                           # generate the dataset
make test                           # 10 tests, all passing
```

Post your `make cloud-check` output in the course channel before Session 1.

---

## What you must finish

Four `TODO` markers are left in the repo deliberately. Each is a graded decision, not busywork.

| Where | What |
|---|---|
| `requirements.txt` | Regenerate with `pip-compile --generate-hashes` |
| `Dockerfile` | Pin the base image by digest; add `--require-hashes` |
| `cloudlayer/<your provider>.py` | Implement `upload`, `download`, `push_image` |
| This README | The reproducibility trade-off question below |

Then:

```bash
make image-push        # image reaches your registry, digest-pinned
dvc init && dvc remote add -d storage ${BLOB_URI}/dvc
dvc add data/raw && dvc push
```

Run five or more tracked runs varying something meaningful — not five identical runs with
different seeds.

---

## Reproducibility trade-off

**I would drop the base-image digest first.** What breaks: `python:3.11-slim` drifts through
patch releases (security fixes, 3.11.x bug fixes), so the OS and interpreter under the model
change. Those rarely move a scikit-learn metric, and if one did, `make verify` would catch it —
the unpinned tag even brings in security fixes. I keep the hashed lock: a swapped package file
changes the build silently. I keep the seed: it costs one line, and without it the split and
trees change every run, moving the metric far beyond the ±0.001 tolerance.

---

## Notes for the grader

- **You do not need cloud access to reproduce.** `make reproduce` regenerates the dataset
  deterministically (`make data`, fixed seed) and trains in Docker; no credentials, no `dvc pull`.
  The DVC remote (Azure Blob, private) holds the same `data/raw` for provenance: version
  `1c886b512c8a5c9bf723da1cd119fc80.dir`, logged on every run as the `dvc_data_version` tag.
- **Cloud:** Azure for Students. Its policy only allows koreacentral / malaysiawest /
  indonesiacentral / centralindia / indiasouthcentral, so storage and ACR live in **malaysiawest**,
  not southeastasia.
- **Lock file** was compiled for the image's target (linux/amd64, Python 3.11) with
  `uv pip compile --generate-hashes`, so it resolves the same on any host OS.
- Developed on Windows; the image builds with `--platform linux/amd64`.

---

## Checklist before you submit

- [ ] `make reproduce` works from a fresh clone, on a machine that is not yours
- [ ] `make verify` passes against your claim line
- [ ] `make test` — all tests pass
- [ ] `make portability-audit` — clean
- [ ] Image builds for `linux/amd64` and is pushed, digest-pinned
- [ ] `dvc push` completed; a grader can `dvc pull`
- [ ] Five or more tracked runs with params, metrics, data fingerprint, and commit SHA
- [ ] Every **REPLACE** block above is gone (the course-materials block at the top stays)
- [ ] `make scan-secrets` is clean

That last check is not optional. A credential in Git history is an automatic deduction in this
course, and rotating it is your responsibility, not the grader's. The scan reads *history*, not
your working tree, because deleting the file is the thing that makes people believe they are
safe. If it finds something, rotate the credential first and rewrite history second — in that
order, because the first one is the only step that actually helps.

---

# Lab 4 — CI/CD, observability and drift

Everything below runs with no cloud account: tests and CI on GitHub Actions, the dashboard
on Prometheus + Grafana in Docker, drift on a scheduled GitHub Actions job. What this means
for the cloud parts of the checklist is stated plainly under **What is not done**.

## Task 1 — Tests that can fail for a real reason

| Category | Where | What fails it |
|---|---|---|
| Unit / split property | `tests/test_data.py` (split tests) | Changing the splitting logic: rows lost or duplicated, a seed with no effect |
| **Data contract** | `tests/test_data.py` (schema, nulls, ranges, target, leakage) | An upstream producer changing the data |
| Model behaviour | `tests/test_model_behaviour.py` | The model changing: a healthy machine flagged, risk falling with wear, latency over budget, constant predictions |
| Integration | `.github/workflows/ci.yml`, job `build` | The serving image not starting, `/predict` not returning a probability and the commit SHA as version |
| Service + metrics | `tests/test_service.py` | Schema validation, readiness, and the five dashboard signals missing from `/metrics` |

**The production incident each data contract test would have caught**

| Test | Incident it catches |
|---|---|
| `test_schema_columns_present_and_typed` | The upstream team renames `temp_c` to `temperature` in a release, or a CSV export turns `load_pct` into a string column. Without this test the model would silently score with a missing feature. |
| `test_no_nulls_in_required_columns` | A sensor gateway drops offline overnight and its readings arrive empty; every machine on that gateway would be scored on imputed garbage. |
| `test_features_within_plausible_ranges` | **Sensor firmware v2 reports temperature in Fahrenheit.** An 80 °C machine reads 176 > 140 and the build stops. This is the bad commit used in Task 3. |
| `test_target_is_binary_and_not_degenerate` | The maintenance system starts writing failure *counts* (2, 3) instead of a 0/1 flag, or a join bug labels every row 0 so the model learns "never fails". |
| `test_identifier_is_unique` | A retried ingest job appends the same batch twice, double-weighting those machines in training. |
| `test_no_machine_leaks_across_splits` | Someone "simplifies" the split to row-wise: the validation score jumps because the model memorises machines, and the real score in production is far lower. |

Model latency budget: 50 ms per prediction, a quarter of the 200 ms endpoint p95 in the SLO
(measured: p50 16.4 ms, p95 17.4 ms per row).

## Task 2 — CI pipeline

`.github/workflows/ci.yml`, on every pull request and every push to main:

```
secret scan (full history, fetch-depth: 0) → lint → portability audit → data contract tests
  → model behaviour tests → service tests → build training + serving images tagged by
  commit SHA → integration test against the running container
```

- The secret scan runs first and on its own: a pushed key cannot be undone by a later commit.
- Images are tagged `itcs355-serve:${{ github.sha }}`, never `latest`.
- The only secret in use, the Discord webhook, lives in the repository secret store as
  `DRIFT_WEBHOOK_URL`; nothing credential-shaped is committed.

## Task 3 — Evidence that a bad commit is blocked

Branch `bad-fahrenheit-feed`: `scripts/make_dataset.py` converts temperature to Fahrenheit,
as a firmware update would. Pull request **[#1](https://github.com/Amzuhe/public_teaching_mlaiops/pull/1)**, closed without merging.

- Failing run: **[CI / test](https://github.com/Amzuhe/public_teaching_mlaiops/actions/runs/38154176441/job/114513606833)**, failed after 44 s at the *Data contract tests* step; `build` was **skipped**, so no image was built or pushed
- Test that caught it: `tests/test_data.py::test_features_within_plausible_ranges`
- Error: `AssertionError: temp_c above plausible ceiling: 240.215` (a 115.7 °C machine read in °F, against the 140 ceiling)

Screenshot: `reports/lab4-blocked-pr.webp`

## Task 4 — Dashboard and SLO

`monitoring/dashboard.json` is a real Grafana dashboard, provisioned by
`monitoring/docker-compose.yml` with Prometheus scraping the service's `/metrics`:

```bash
uvicorn service.app:app --port 8080            # with MODEL_PATH set, see `make serve`
docker compose -f monitoring/docker-compose.yml up -d
python scripts/traffic.py --n 600              # healthy traffic, ~2% malformed requests
python scripts/traffic.py --n 600 --shift 4    # drifted traffic
# http://localhost:3000
```

Panels: request rate · error rate split 4xx / 5xx · latency p50 / p95 / p99 against the 200 ms
SLO line · **rolling PSI of every feature over the last 500 inputs** against the 0.08 alert
line · rolling mean of `temp_c` · model version in production. Screenshot during the
injected drift: `reports/lab4-dashboard-drift.webp` (healthy before 22:58, +4 °C traffic after).

The SLOs, each with target, window and what happens when the budget is spent, are in
`monitoring/slo.yaml`: availability 99.5% / 30 days (deploy freeze, roll back to the last good
SHA), latency p95 < 200 ms / 7 days (stop shipping slower models, profile, roll back),
model age ≤ 30 days (retrain only after contract tests and the drift check pass on the new data).

## Task 5 — Drift detection

**Statistic:** PSI per feature (with KS reported alongside), `monitoring/drift.py`.
**Schedule:** `.github/workflows/drift.yml`, hourly, plus on every new production window.
**Channel:** Discord, through the `DRIFT_WEBHOOK_URL` repository secret.

**Threshold: PSI ≥ 0.08 on any feature, over a 500-reading window sampled across the fleet.**
Measured, not copied:

| 500-reading window | PSI / how often it alerts |
|---|---|
| Healthy data, 300 resampled windows, worst feature | median 0.028, p99 0.048, max **0.054** → alerts 0% of the time |
| `temp_c` +2 °C | alerts 9% |
| `temp_c` +3 °C (a third of a standard deviation) | alerts **87%** |
| `temp_c` +4 °C | alerts **100%** |
| `temp_c` +6 °C | 0.38, the only shift the textbook 0.25 catches |

So 0.08 sits clear of normal noise (no false alarms) yet catches a 3–4 °C shift, which is
when the model's predictions start to move (see the post-mortem). 0.25 would wait until +6 °C.

Two things the calibration showed:
- **The threshold is tied to the window size.** At 250 readings healthy noise reaches p99 0.096,
  so a smaller window needs a higher threshold. The dashboard shows this live: rolling PSI
  starts above 0.1 while the window is filling and settles under 0.08 once it reaches 500.
- **The window must cover the fleet.** The data is stored machine by machine; the last 500 rows
  are only 20 of the 240 machines, and that fleet-mix difference alone scored as drift on
  healthy data. The detector samples across all rows.

## Task 6 — Injected drift

Injection: `python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 4`, the
shifted window committed as `data/current.csv` and pushed.

| | Time (UTC) |
|---|---|
| Drift injected | 2026-10-11 15:57:36 |
| Alert in Discord | 2026-10-11 15:58:17 |
| **Detection time** | **41 seconds** (push-triggered run; on the hourly schedule alone it would be up to 60 minutes) |

Alert: `temp_c PSI 0.124 (threshold 0.08), mean 79.58 -> 82.98` (the window is a 500-reading sample, hence 82.98 rather than the full-file 83.58); the other five features stayed
below 0.08. Screenshots: `reports/lab4-drift-alert.png`, `reports/lab4-dashboard-drift.webp`.

**Post-mortem**

```
What fired: the hourly drift check, temp_c PSI 0.124 against a 0.08 threshold; no other feature moved.
True cause: every temp_c reading rose by exactly 4 °C while vibration, load and failures did not move (schema and null rate unchanged) — a uniform offset like this is a sensor calibration or firmware fault, not machines really running hotter.
Retrain, roll back, or no action — and why: no action on the model: it has not changed, so there is nothing to roll back, and retraining would teach it the faulty offset as normal; fix or re-calibrate the temperature feed with its owner and backfill the affected readings.
What this would have cost if unnoticed for a week: the model would flag 19% more readings at p ≥ 0.3 (13.0% → 15.4%) and 13% more at p ≥ 0.5, so roughly one in six inspection work orders that week would be for machines that were fine.
How to prevent or detect it faster: add an ingest-time contract on the fleet mean of each feature (temp_c within ±2 °C of the reference) so an offset is rejected at the source in minutes, and check sensor firmware changes against the data contract before rollout.
```

## What is not done (and why)

No cloud account was set up for this lab, so these parts of the checklist are not met:
pushing the image to a cloud registry, CD to a staging endpoint, OIDC federation, the managed
endpoint, and a cloud scheduler. `.github/workflows/cd.yml` still holds the course placeholders.
In their place: images are built and integration-tested per commit SHA in CI, and the drift
schedule runs on GitHub Actions cron.

**Teardown:** nothing runs in a cloud. Locally: `docker compose -f monitoring/docker-compose.yml down`
and stop the uvicorn process. The scheduled drift workflow can be disabled from the Actions tab.
