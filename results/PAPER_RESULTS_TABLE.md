# MaSE-Net results table (draft for paper / report)

Last local update: 2026-07-27. Full-data cells filled from professor runs when JSON arrives.

## Accuracy

| Method | 5% test | Full test | Mixture weights | Notes |
|--------|---------|-----------|-----------------|-------|
| Keras DeepYeastNet | — | ~88.4% | — | published baseline |
| MaSE Lite **v2** | **85.03%** | **89.1%** | fixed 0.25×4 | `--legacy-v2-loss` |
| MaSE Lite **v3** | ~86.1% (abl_min05) | **87.75%** | collapsed ~0.78 | learnable, weak floor |
| MaSE Lite **v4** | **85.87%** | **≈89.58%** | frozen 0.25×4 | **accuracy SOTA** (train, no TTA) |
| MaSE Lite **v5** | — | **89.16%** | [0.16, 0.16, 0.44, 0.24] | learnable; UAUC(PE)=0.9249; below v4 acc |
| MaSE Lite **v8** | — | **89.05%** / TTA **89.69%** | sample-wise w(x) | ID-Gate; >P1, below P2.5+TTA 89.82% |
| MaSE Lite **SFRM** | **85.09%** / vote **84.27%** | — | frozen 0.25×4 | bypass selector, S=24; vote hurt |
| MaSE Lite **SFRM+sel** | **85.71%** | — | frozen 0.25×4 | keep selector, S=16; −0.16pp vs v4 |

**5% decision rule for promoting v5 to full-data:** test > 85.87% **and** no collapse (`min_w≥0.10`, `max_w≤0.55`). Else keep recommending **v4** for full data.

## Uncertainty (primary = AUROC of predictive entropy)

| Checkpoint | Split | AUROC(PE) = UAUC | Softmax (1−maxprob) AUROC | Source |
|------------|-------|------------------|---------------------------|--------|
| 5% v4 `best.pt` | test | *running eval* | *running* | `.../v4_default/uq_mc_dropout/` |
| 5% v5 `best.pt` | test | after train | after train | embedded `--with-uq` |
| Full v4 | test | TBD | TBD | ask professor for `uq_mc_dropout/summary.json` |
| Full v5 | test | **0.9249** | — | professor run summary |

Do **not** treat Softmax `UAUC_via_1_minus_maxprob` as the primary UQ claim.

## Sources (5%)

- v2 5%: `results/v2/5pct_results.json`
- v4 5%: `results/v4/5pct_results.json` (test 0.8587)
- SFRM 5%: `results/sfrm/5pct_results.json` (unmasked 0.8509); `results/sfrm/5pct_vote_summary.json` (vote 0.8427)
- SFRM+sel 5%: `results/sfrm/5pct_sel_results.json` (test 0.8571)
- Ablation ranking: `results/full_friendly_ablation_summary.jsonl`
- Full-data numbers: see accuracy table above / [FINE_TUNING_V4_TO_V7.md](../FINE_TUNING_V4_TO_V7.md)
