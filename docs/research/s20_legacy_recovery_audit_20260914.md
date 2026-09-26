# Legacy exact-state recovery audit

Date: 2026-09-14. Scope: read-only recovery investigation; no fits, source modification, promotion or production action.

## Recovery evidence

The original producer is tracked at commit `3368b6b` (`research: validate R20-anchored S20 ranking`). Both the current script and that committed version save only residual and ranker models for the confirmation run. The commit tree does not contain the confirmation model state bundle. Current `output/experiments/s20_20r_confirmation/run_v1` contains predictions, reports, three residual models and three ranker models, but no final anchor or R20 reference/calibration state.

An initial rg enumeration did not expose all ignored model assets, so native filesystem enumeration was used to inspect the relevant experiment directories. It found possible names, not recoverable exact-run identities:

| Located artifacts | What they establish | Why they cannot replace confirmation states |
|---|---|---|
| `r20_target_prob_v2/wf1..wf3/ordinal_multiclass.txt`, `calibrators.json` | Earlier fold models and calibrators exist | No provenance binding to confirmation's separately fitted ordinal/binary models or its calibration segment |
| `s20_20r_residual_portable/wf1..wf3/r20_anchor_oof_early.txt`, `r20_anchor_refit.txt` | Portable development anchors were saved | Development folds are not the final confirmation refit; shared feature count/contract name is not model identity |
| `s20_20r_residual`, `s20_20r_residual_smoke`, `s20_20r_residual_v11` anchors | Other trial variants exist | Different trial/fold or feature scope; no exact confirmation binding |

Search is bounded to local relevant experiment outputs and the producing commit, not a claim about external backups or all Git objects. Do not compare candidate predictions to pick whichever looks closest and call that identity recovery. No exact-state replacement was selected.

## Additional training-dependency finding

In `scripts/confirm_s20_20r_portable.py`, the definitions are:

```text
mature := horizon_end_date < 20260127
final_anchor_fit := development[mature]
residual_tune := development[20251101 <= trade_date <= 20260126,
                             mature, positive20 >= 0]
final_anchor := refit(final_anchor_fit)
residual_tune.r20_anchor := final_anchor.predict(residual_tune)
```

Consequently `residual_tune` is a subset of `final_anchor_fit` by construction. Its anchor predictions are not out-of-fit predictions. This is a code-level dependency proof, not an independently enumerated row-count audit. It can make tuning inputs optimistic; it does not alone prove the later confirmation labels entered fitting. The confirmation period is already exposed development evidence in the current protocol regardless.

Any new reconstruction trial must distinguish exact legacy reproduction from corrected training. A corrected pipeline needs chronological/cross-fitted anchor predictions for downstream training/tuning/calibration/selection, with an explicit membership audit; its results belong to a new trial, not the archived run. Do not change the archived recipe silently and claim the old metric reproduced.

## Next execution decision

Exact legacy-state recovery remains unresolved. Saved-ranking metrics and composition remain reproducible, and the two v3 models have full saved-inference reproduction. Keep these evidence levels separate in H03. Continue independent raw-label/new-label alignment and H03 implementation; if original state cannot be recovered, register a bounded reconstruction after data gates, including every anchor/refit in the fit budget. No formal H03 acceptance from this audit.
