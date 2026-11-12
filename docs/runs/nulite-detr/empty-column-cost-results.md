# Empty-column matcher cost — results (2026-11-11)

Single-mechanism run: `main/pannuke-thin-t12-empty-cost.yaml` vs
`main/pannuke-thin-t12.yaml`. One key differs (`detr_matcher_empty_cost`),
`lambda` = 1.0, same 600-epoch schedule stopped at 332, same fold1 oracle.

Diagnostic, not a paper claim. One fold, one seed.

## Run

| item | value |
|---|---|
| run dir | `output/v2/train57_empty-cost` |
| wall time | 23,828.8 s = **6.62 h** (71.8 s/epoch) |
| epochs | 332 / 332 |
| final validator bPQ (trajectory only) | 0.66013 vs control 0.66005 |
| params | 9.55M (unchanged) |

The validator curve crossed the control repeatedly (ahead at ep 100–150,
behind at 250, ahead at 265), i.e. seed-noise shaped. Only the held-out
evaluation is decisive.

## Held-out Fold3 (`eval_lsp`, 2,722 images, 66,654 GT, raw `nc+1` argmax, no NMS)

Control `train52_fastvit-t12` was re-evaluated in the same session and
reproduced its recorded numbers exactly.

| metric | control | empty-cost (last) | delta | empty-cost (best) |
|---|---:|---:|---:|---:|
| bPQ | 0.6372 | **0.6391** | **+0.0019** | 0.6384 |
| bDQ | 0.7798 | 0.7832 | +0.0034 | 0.7829 |
| bSQ | 0.8032 | 0.8037 | +0.0005 | 0.8021 |
| mPQ | 0.4561 | 0.4597 | +0.0036 | 0.4599 |
| macro bPQ | 0.6566 | 0.6578 | +0.0012 | 0.6571 |
| macro mPQ | 0.4618 | 0.4609 | −0.0009 | 0.4618 |
| F1 (centroid r=12) | 0.7987 | 0.8014 | +0.0027 | 0.8008 |
| precision | 0.8270 | 0.8256 | −0.0014 | 0.8232 |
| recall | 0.7724 | 0.7786 | +0.0062 | 0.7796 |

`last.pt` beats `best.pt` on bPQ, bDQ, macro bPQ and F1, so `last` is the
reported checkpoint and the choice is not cherry-picking.

## Cohort audit (384 images, 8,917 GT)

| cohort | control | empty-cost | delta |
|---|---:|---:|---:|
| matched | 6,929 | 7,031 | **+102** |
| good_suppressed_bg | 1,163 | 1,045 | **−118** |
| no_good_polygon | 798 | 817 | +19 |
| wrong_type | 974 | 1,024 | +50 |
| unmatched_predictions | 1,262 | 1,258 | −4 |
| duplicate_like_raw | 15 | 6 | −9 |
| good_fg_unmatched | 27 | 24 | −3 |
| raw predictions | 8,196 | 8,290 | +94 |

This is the first matcher experiment whose cohort movement matches its design:
the target cohort (`good_suppressed_bg`, well-placed queries that claimed
empty) fell 118, matched rose 102, and — unlike the soft inside gate — the
freed nuclei did **not** turn into badly segmented polygons
(`no_good_polygon` +19 only, versus +30 there) or into extra output
(`unmatched_predictions` flat, versus +73 there).

## Verdict against the pre-registered rule

The rule was: adopt only if bPQ **and** macro bPQ both gain > +0.002.

- bPQ +0.0019 — **just below** the bar.
- macro bPQ +0.0012 — **below** the bar.
- Reject line (bPQ −0.002) not approached.

So, formally: **not adopted**. It is the first directionally positive matcher
change — every aggregate that depends on finding nuclei improved (bPQ, bDQ,
mPQ, F1, recall) and the intended cohort moved — but it does not clear the
pre-registered effect size, and macro mPQ is very slightly negative. Reporting
it as a win would be exactly the kind of threshold-shopping the rule exists to
prevent.

## What this does and does not establish

- Establishes: the empty decision is *a* real constraint (the target cohort
  responds), and penalising it in the matcher is a valid way to reach it.
- Does not establish: that it is *the* binding constraint. A ~1.7% cohort shift
  producing +0.0019 bPQ suggests the ceiling of this lever is around the bar,
  not far above it.
- The remaining structure of the failure is unchanged: `wrong_type` actually
  increased (+50), and the class-vs-class margin is still zero
  (`p(correct) − p(best_wrong)` median 0.000). No matcher cost can fix type
  confusion.

## Artefacts

`output/evals/empty_cost/`: `ctrl.log`, `empty.log`, `best.log`, `ctrl.npz`,
`empty.npz`, `ctrl_audit.log`, `empty_audit.log`, `ctrl_cohort.csv/json`,
`empty_cohort.csv/json`, `run.sh`, `run_best.sh`, `driver.log`,
`best_driver.log`.

## Follow-ups, in order of value

1. **`lambda` sweep** (2.0, 4.0) at fixed fold — the mechanism works in the
   right direction and 1.0 was frozen a priori; 2× and 4× are the obvious
   next points, ~6.5 h each.
2. **Combine with a classification-separation lever** — `wrong_type` grew, and
   the type margin is zero, so matcher work alone is capped.
3. Only after a variant clears the bar: 3-fold rotation for a literature claim.

## Follow-up run: `lambda = 2.0` (staged, not yet trained)

The user chose a single point, `lambda = 2.0`, rather than the two-point sweep.

| run | recipe | diff vs control | diff vs `lambda=1` |
|---|---|---|---|
| `train57_empty-cost` | `main/pannuke-thin-t12-empty-cost.yaml` | one key | — |
| `train58_empty-cost-l2` | `main/pannuke-thin-t12-empty-cost-l2.yaml` | two keys (`empty_cost`, `lambda`) | one key (`lambda`) |

Verified by parsing all three recipes with `ETLConfig`: `lambda=1` and
`lambda=2` differ in exactly `detr_matcher_empty_cost_lambda (1.0 -> 2.0)`, so
the dose-response comparison with the trained `lambda=1` arm is clean. Against
the control the arm changes two keys, but they are one mechanism at a chosen
strength.

Runner: `scripts/train_empty_cost_l2.sh` (single run, `--epochs 600 --end 332`,
~6.5 h).

Decision rule is unchanged from the pre-registration: adopt only if bPQ AND
macro bPQ both gain > +0.002 over the control (0.6372 / 0.6566); reject if bPQ
drops > 0.002. Additional stopping rule for the sweep: if `lambda=2` reverses
the `lambda=1` direction (bPQ below control), stop — the lever is not
dose-monotone and a further increase is not justified. Report
`good_suppressed_bg`, `no_good_polygon`, `wrong_type`, and
`unmatched_predictions` in every case; the diagnostic question is whether the
remaining 1,045 suppressed nuclei yield more matches or start converting into
`no_good_polygon`.

## Result: `lambda = 2.0` — rejected, sweep closed

`output/v2/train58_empty-cost-l2`: 332/332 epochs, 23,851 s (6.63 h, 71.8
s/epoch). Final validator bPQ 0.65739 (below both control 0.66005 and
`lambda=1` 0.66013).

### Held-out Fold3 (2,722 images, 66,654 GT, raw `nc+1` argmax, no NMS)

Control reproduced exactly again.

| metric | control | `lambda=1` | `lambda=2` | Δ vs control |
|---|---:|---:|---:|---:|
| bPQ | 0.6372 | 0.6391 | 0.6360 | **−0.0012** |
| bDQ | 0.7798 | 0.7832 | 0.7797 | −0.0001 |
| bSQ | 0.8032 | 0.8037 | 0.8026 | −0.0006 |
| mPQ | 0.4561 | 0.4597 | 0.4578 | +0.0017 |
| macro bPQ | 0.6566 | 0.6578 | 0.6554 | **−0.0012** |
| macro mPQ | 0.4618 | 0.4609 | 0.4617 | −0.0001 |
| F1 | 0.7987 | 0.8014 | 0.7963 | −0.0024 |
| P | 0.8270 | 0.8256 | 0.8234 | −0.0036 |
| R | 0.7724 | 0.7786 | 0.7708 | −0.0016 |

### Cohort audit (384 images, 8,917 GT)

| cohort | control | `lambda=1` | `lambda=2` |
|---|---:|---:|---:|
| matched | 6,929 | 7,031 | 6,940 |
| good_suppressed_bg | 1,163 | 1,045 | 1,151 |
| no_good_polygon | 798 | 817 | 802 |
| wrong_type | 974 | 1,024 | 1,011 |
| unmatched_predictions | 1,262 | 1,258 | 1,261 |
| duplicate_like_raw | 15 | 6 | 4 |

### Reading

The dose response is non-monotone: `lambda=1` gained +0.0019 bPQ and freed 118
suppressed nuclei; `lambda=2` gives back all of it (bPQ −0.0012, suppression
only −12). Doubling the weight does not buy more suppression — it mostly adds
`wrong_type` instances (+37) and loses precision (−0.0036) without converting
suppressed nuclei into matches.

This is the predicted failure mode of a row-constant term: it can only change
which queries are left idle, so past a point it stops selecting better rows and
starts displacing rows that were already the best available. `good_suppressed_bg`
remaining at 1,151 under `lambda=2` (vs 1,045 at `lambda=1`) means the extra
weight is not even landing on the intended cohort.

The pre-registered stopping rule ("stop if `lambda=2` falls below control on
bPQ") fires. **The sweep is closed: `lambda=1` is the best point, and it is
still below the adoption bar (+0.0019 and +0.0012 vs the required +0.002).**
