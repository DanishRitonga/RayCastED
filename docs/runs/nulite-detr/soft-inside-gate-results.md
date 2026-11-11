# Soft inside gate — results (2026-11-11)

Follow-up to `soft-inside-gate-plan.md`. Single fold, single seed, T12, trained
`--epochs 600 --end 332` exactly like the T8/T12 grids.

**Verdict: rejected.** bPQ fell 0.0027, past the pre-registered −0.002 reject
line, and macro bPQ fell 0.0018. The mechanism did what it was designed to do
(fewer background-suppressed nuclei) but the gain was swamped by a larger rise
in poor-polygon assignments.

## Provenance

- Evaluator: `raycasted.evals.eval_lsp`, held-out Fold3 parquet (2,722 images,
  66,654 GT), raw `nc+1` all-class argmax, no threshold, no NMS.
- Checkpoints: `last.pt` for the reported number; `best.pt` evaluated only as a
  diagnostic (worse on every headline metric, so nothing was cherry-picked).
- Control: `train52_fastvit-t12`, evaluated in the same session.
- Treatment: `train56_soft-inside`, `main/pannuke-thin-t12-soft-inside.yaml`,
  differs from the control in exactly one key (`detr_matcher_soft_inside`).
- Frozen calibration: `w=20`, `p=2`, chosen before any evaluation and not
  re-tuned after probing.

## Training

332/332 epochs, 23,537.7 s cumulative = **6.54 h**, 70.9 s/epoch — same as the
T12 base (70.9 s/epoch), confirming the matcher change is loss-only and adds no
step cost.

Validator final (trajectory signal only): bPQ 0.65591 / bDQ 0.80551, the lowest
bPQ in the T12 family (base 0.66005, range across the grid 0.65598–0.66198).

## Held-out Fold3

| metric | control | soft-inside | delta |
|---|---:|---:|---:|
| bPQ | 0.6372 | 0.6345 | **−0.0027** |
| bDQ | 0.7798 | 0.7775 | −0.0023 |
| bSQ | 0.8032 | 0.8034 | +0.0002 |
| mPQ | 0.4561 | 0.4571 | +0.0010 |
| macro bPQ | 0.6566 | 0.6548 | **−0.0018** |
| macro mPQ | 0.4618 | 0.4605 | −0.0013 |
| F1 (centroid r=12) | 0.7987 | 0.7957 | −0.0030 |
| precision | 0.8270 | 0.8197 | −0.0073 |
| recall | 0.7724 | 0.7730 | +0.0006 |
| params | 9.55M | 9.55M | 0 |

`best.pt` diagnostic: bPQ 0.6340, bDQ 0.7775, bSQ 0.8021, mPQ 0.4563, macro
bPQ 0.6527, macro mPQ 0.4592, F1 0.7938, P 0.8140, R 0.7746.

## Cohort breakdown (384 images, 8,917 GT)

| cohort | control | soft-inside | delta |
|---|---:|---:|---:|
| matched | 6,929 | 6,903 | −26 |
| `good_suppressed_bg` | 1,163 | 1,149 | **−14** |
| `no_good_polygon` | 798 | 828 | **+30** |
| `wrong_type` | 974 | 938 | −36 |
| `unmatched_predictions` | 1,262 | 1,335 | **+73** |
| `duplicate_like_raw_predictions` | 15 | 19 | +4 |
| `good_fg_unmatched` | 27 | 37 | +10 |
| raw predictions | 8,196 | 8,244 | +48 |

## Reading

The intervention targeted exactly the right cohort and moved it the right way:
`good_suppressed_bg` dropped 14. But two larger effects offset it:

- `no_good_polygon` rose 30. A GT whose inside candidates were all poor now
  gets assigned to a near-boundary outside query instead of being suppressed —
  so the nucleus is "found" but with a polygon too bad to count. That is a
  wash at best and a precision loss in practice.
- `unmatched_predictions` rose 73 and precision fell 0.0073. With the veto
  softened, more queries are viable competitors, so more marginally-placed
  queries survive the assignment as unmatched output.

Net `matched` fell by 26. Recall is flat (+0.0006), so the change bought no
detection; it only redistributed failures.

This is consistent with the pre-run probe reading: the soft gate lets
near-boundary pairs compete whenever `excess` is small, and the near-0 tail of
`dist/d_phi − 1` is dense. The probe also showed the flip rate was nearly
independent of `w` (28.6% at w=2, 25.4% at w=1000), i.e. the change is not
really a tunable "how soft" knob — it is a structural change in which query a
GT is allowed to pick, and that structural change is net negative.

## What this rules out

The inside gate is not the binding constraint. It was the leading geometric
hypothesis after the empty-column analysis, and it is now measured: relaxing it
trades suppression for bad polygons and extra output.

Remaining untested levers in this family, in the order they now seem worth
trying:

1. **Empty-column cost** — the matcher still scores `p_scores[:, g_cls]`
   without comparing against the no-object column, so a confident-empty query
   is not explicitly preferred when a GT should simply be suppressed.
2. **Allowing GTs to stay unmatched** — currently a GT must take some query;
   there is no way for the assignment to say "nothing here is good enough".

Both are changes to *which decision is represented*, not to the distance
threshold, which is what this experiment showed to be inert.

## Artefacts

- `output/evals/soft_inside/{last,best}.log` — full evaluator output.
- `output/evals/soft_inside/{ctrl,soft}.npz` + `{ctrl,soft}_audit.log` — raw
  query caches and cohort audits for the 384-image paired comparison.
- `output/v2/train56_soft-inside/` — run directory.
- `scripts/train_soft_inside.sh`, `main/pannuke-thin-t12-soft-inside.yaml`.
