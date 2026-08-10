# MaSE Lite result snapshots

JSON results organized by version (synced from `deepyeast-workspace`).

```text
results/
  v2/
    5pct_results.json          # 5% pilot, MaSE Lite v2
  v4/
    5pct_results.json          # 5% pilot, MaSE Lite v4 (frozen w)
    full_results.json          # full data, frozen w=0.25 (terminal archive)
  v5/
    full_results.json          # full data, learnable w + UQ (terminal archive)
  v6/
    phase1_results.json        # full data, TP-AHF Phase 1 (= v4 recipe)
    phase2_results.json        # full data, ensemble_logits only (terminal archive)
    phase25_results.json       # full data, Phase 2.5 (heads + w)
    phase25_tta_summary.json   # TTA on Phase 2.5 best.pt
  v7/
    results.json               # full data, tuned Phase 2.5
    tta_summary.json           # TTA on v7 best.pt
  uq_5pct/  uq_10pct/  head_contrib_10pct/  figs/
```

## Full-data accuracy (seed=42)

| Version | File | Test |
|---------|------|------|
| **v4** | `v4/full_results.json` | **≈89.58%** |
| v5 | `v5/full_results.json` | ~89.16% |
| v6 Phase 1 | `v6/phase1_results.json` | ~88.95% |
| v6 Phase 2 | `v6/phase2_results.json` | ~89.01% |
| v6 Phase 2.5 | `v6/phase25_results.json` | ~89.07% |
| v6 Phase 2.5 + TTA | `v6/phase25_tta_summary.json` → `test_tta_accuracy` | ~89.82% |
| v7 | `v7/results.json` | ~89.06% |
| v7 + TTA | `v7/tta_summary.json` → `test_tta_accuracy` | ~89.82% |

`v4/full_results.json`, `v5/full_results.json`, and `v6/phase2_results.json` are **terminal-summary archives** (`source: professor_terminal_summary`) because the original trainer `results.json` files were not recovered from the professor machine. Other files are raw trainer / eval dumps.

Other folders (`uq_5pct/`, `uq_10pct/`, `head_contrib_10pct/`, `figs/`) stay as-is.
