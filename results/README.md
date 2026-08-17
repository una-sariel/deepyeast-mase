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
  v9/
    results.json               # full data, RSB train (random mask)
    rsb_summary.json           # R=16 vote vs learned-mask ref
  v10/
    5pct_results.json          # 5% v10 SFRM (shared-region, selector bypass)
    5pct_vote_summary.json     # 7-window vote eval
    5pct_sel_results.json      # 5% v10 + keep selector, S=16
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
| v9 random-mask train | 84.87% | `v9/results.json` |
| v9 RSB (R=16) | 86.28% | `v9/rsb_summary.json` |
| v9 learned-mask ref | 88.96% | `v9/rsb_summary.json` |

## 5% accuracy (seed=42)

| Version | Val | Test | JSON |
|---------|-----|------|------|
| v4 frozen | 82.7% | **85.87%** | `v4/5pct_results.json` |
| v10 SFRM bypass S=24 | 83.22% | 85.09% | `v10/5pct_results.json` |
| v10 7-window vote | — | 84.27% | `v10/5pct_vote_summary.json` |
| v10 + selector S=16 | 83.68% | 85.71% | `v10/5pct_sel_results.json` |

Other folders (`uq_5pct/`, `uq_10pct/`, `head_contrib_10pct/`, `figs/`) stay as-is.
