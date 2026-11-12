# Conditional empty-column cost — results

Status: **evaluated and rejected.**
Run: `output/v2/train59_empty-cond/weights/last.pt` (`train59_empty-cond`).
Plan: `docs/runs/nulite-detr/empty-column-cost-conditional-plan.md`.
Protocol: `raycasted/evals/eval_lsp.py`, held-out Fold3 parquet, 2,722 images /
66,654 GT, raw `nc+1` argmax, no confidence threshold, no NMS, polygon overlap
resolved by greatest Euclidean interior distance.

## Run

| field | value |
|---|---|
| recipe | `main/pannuke-thin-t12-empty-cost-cond.yaml` |
| diff vs λ=1 arm | `detr_matcher_empty_cost_mode: additive → conditional` (one key) |
| epochs | 332/332 (`--epochs 600 --end 332`) |
| wall | 23,728 s = **6.59 h** |
| s/epoch | 71.5 |
| final validator bPQ (trajectory only) | 0.65981 (control 0.66005, λ=1 0.66013, λ=2 0.65739) |

## Held-out Fold3

Control re-evaluated in-session and reproduced the recorded numbers exactly.

| metric | control | conditional (last) | delta | conditional (best) |
|---|---:|---:|---:|---:|
| bPQ | 0.6372 | **0.6365** | **−0.0007** | 0.6325 |
| bDQ | 0.7798 | 0.7778 | −0.0020 | 0.7782 |
| bSQ | 0.8032 | 0.8045 | +0.0013 | 0.8004 |
| mPQ | 0.4561 | 0.4553 | −0.0008 | 0.4517 |
| macro bPQ | 0.6566 | 0.6565 | −0.0001 | 0.6529 |
| macro mPQ | 0.4618 | 0.4567 | **−0.0051** | 0.4553 |
| F1 | 0.7987 | 0.7989 | +0.0002 | 0.7946 |
| P | 0.8270 | 0.8302 | +0.0032 | 0.8199 |
| R | 0.7724 | 0.7698 | −0.0026 | 0.7709 |
| raw pred instances | 62,263 | 61,805 | −458 | 62,692 |

`last.pt` beats `best.pt` on bPQ, bSQ, mPQ, macro bPQ, macro mPQ and F1, so
reporting `last` is not cherry-picking.

## Cohort audit (384 images, 8,917 GT)

| cohort | control | conditional | delta | (λ=1 arm, for reference) |
|---|---:|---:|---:|---:|
| matched | 6,929 | 6,889 | **−40** | 7,031 (+102) |
| good_suppressed_bg | 1,163 | 1,241 | **+78** | 1,045 (−118) |
| no_good_polygon | 798 | 764 | −34 | 817 (+19) |
| wrong_type | 974 | 988 | +14 | 1,024 (+50) |
| unmatched_predictions | 1,262 | 1,214 | −48 | 1,258 (−4) |
| duplicate_like_raw | 15 | 6 | −9 | 6 (−9) |
| good_fg_unmatched | 27 | 23 | −4 | 24 (−3) |
| raw_predictions | 8,196 | 8,103 | −93 | 8,290 (+94) |

## Verdict

Against the pre-registered rule (adopt only if bPQ **and** macro bPQ both gain
> +0.002; reject if bPQ drops > 0.002): **no adoption** — both are flat to
slightly negative (bPQ −0.0007, macro bPQ −0.0001), and the arm it was designed
to improve, the λ=1 additive run (bPQ 0.6391, macro 0.6578), is clearly better.

The mechanism did what the algebra predicts, but on the wrong axis:

- Renormalising over `{class, empty}` makes the class cost **scale-free**, so the
  matcher stops rewarding absolute foreground confidence and starts preferring
  queries with the highest foreground/empty *ratio*. Output count falls
  (62,263 → 61,805) and precision rises (+0.0032) at the cost of recall
  (−0.0026) — a pure confidence-thresholding effect expressed through the cost
  matrix rather than at inference, and it caps PQ.
- The designed cohort moved the **wrong way**: `good_suppressed_bg` 1,163 → 1,241
  (+78), against 1,045 (−118) for the additive arm. A nucleus whose best query
  is "not empty but not confident" now looks actively bad instead of merely
  unremarkable, so it stays unmatched.
- Macro mPQ, the worst metric (−0.0051), confirms the ratio mostly reshuffled
  which queries were spent rather than improving their polygons.

## What this establishes

The empty-column signal is real (`+102` matched under the additive form) but its
useful shape is the **additive, row-constant** one. Making it per-GT via
renormalisation destroys the absolute-confidence term that was doing the work
and turns a recall-side fix into a precision-side one. There is nothing left to
tune along this axis: additive was swept at λ = 1 and 2 (λ=2 reverses), and the
conditional form is now measured at its single natural strength.

The remaining headroom is elsewhere: `wrong_type` (974/8,917) is untouched by
any matcher cost because `p(correct) − p(best_wrong)` has median 0.000 on this
head, and the good-polygon/background suppression cohort still holds ~1,045
nuclei under the best matcher arm.

## Artefacts

`output/evals/empty_cond/`: `ctrl.log`, `cond.log`, `condb.log`,
`ctrl.npz` / `cond.npz` / `condb.npz` (prediction caches),
`ctrl_audit.log`, `cond_audit.log`, `condb_audit.log`,
`*_cohort.csv` / `*_cohort.json` (per-instance and summary cohort records),
`run.sh`, `driver.log` (all exits 0, `ALL_DONE`).
