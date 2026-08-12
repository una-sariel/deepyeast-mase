# MaSE Lite result snapshots

JSON results organized by version (synced from `deepyeast-workspace`).

```text
results/
  v2/
    5pct_results.json          # 5% pilot, MaSE Lite v2
  v4/
    5pct_results.json          # 5% pilot, MaSE Lite v4 (frozen w)
  v6/
    phase1_results.json        # full data, TP-AHF Phase 1 (= v4 recipe)
    phase25_results.json       # full data, Phase 2.5 (heads + w)
    phase25_tta_summary.json   # TTA on Phase 2.5 best.pt
  v7/
    results.json               # full data, tuned Phase 2.5
    tta_summary.json           # TTA on v7 best.pt
  v8/
    results.json               # full data, ID-Gate
    tta_summary.json           # TTA on v8 best.pt
  uq_5pct/  uq_10pct/  head_contrib_10pct/  figs/
```

## Full-data accuracy (seed=42)

| Version | Test | JSON |
|---------|------|------|
| **v4** | **≈89.58%** | — |
| v5 | ~89.16% | — |
| v6 Phase 1 | ~88.95% | `v6/phase1_results.json` |
| v6 Phase 2 | ~89.01% | — |
| v6 Phase 2.5 | ~89.07% | `v6/phase25_results.json` |
| **v6 Phase 2.5 + TTA** | **~89.82%** | `v6/phase25_tta_summary.json` |
| v7 | ~89.06% | `v7/results.json` |
| **v7 + TTA** | **~89.82%** | `v7/tta_summary.json` |
| v8 ID-Gate | ~89.05% | `v8/results.json` |
| v8 + TTA | ~89.69% | `v8/tta_summary.json` |

Other folders (`uq_5pct/`, `uq_10pct/`, `head_contrib_10pct/`, `figs/`) stay as-is.
