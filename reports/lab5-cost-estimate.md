# Lab 5: cost estimate, made BEFORE running (Task 5, item 1)

Committed on 2026-10-23, before any Lab 5 pipeline run. The actual figure is compared against
this in `reports/lab5-cost.md`.

Subscription: Azure for Students, region `malaysiawest`, resource group `itcs355-6688003`,
everything tagged `course=itcs355`. Rates are from `src/costs.py` (USD x 36).

| Component | What runs | Estimate (THB) | Reasoning |
|---|---|---|---|
| Registry | ACR Basic, already running | 42 | ~$0.167/day x 7 days x 36 |
| Training / pipeline | `Standard_B2s`, ~2 h of runs | 3 | 1.50 THB/h x 2 h (spot not available on B-series) |
| Storage | Blob, < 100 MB | 1 | data + model artifacts, negligible |
| Serving | Container Apps, scale-to-zero | 0 | inside the monthly free grant at lab traffic |
| Monitoring | Log Analytics, < 1 GB | 0 | inside the free 5 GB/month |
| **Total for the Lab 5 week** | | **~46 THB** | |

Biggest risk to this estimate: the **registry**. It bills per day whether or not anything is
pushed, so the estimate is mostly a function of how many days the resource group stays alive,
not of how much work is done. Second risk: anything left running overnight.
