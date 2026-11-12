# Conditional empty-column cost — plan

Status: **implemented, not yet trained.**
Control: `main/pannuke-thin-t12.yaml` (`train52_fastvit-t12`, held-out Fold3
bPQ 0.6372 / macro bPQ 0.6566).
Nearest arm: `main/pannuke-thin-t12-empty-cost.yaml` (`train57_empty-cost`,
bPQ 0.6391 / macro 0.6578).

## Why

Two matcher levers have already been tested on this codebase:

| lever | mechanism | held-out bPQ | verdict |
|---|---|---:|---|
| soft inside gate | `w * excess**p` instead of the 9999 veto (geometry) | 0.6345 (−0.0027) | rejected |
| additive empty cost | `cost += λ * −log(1 − p_empty)` (logits, λ=1) | 0.6391 (+0.0019) | below the +0.002 bar |
| additive empty cost, λ=2 | same, heavier | 0.6360 (−0.0012) | rejected, sweep closed |

The λ sweep is non-monotone: λ=1 is the best point and λ=2 gives it all back.
That shape is expected for a term that is **row-constant** — `−log(1 − p_empty)`
is identical for every GT column of a query, so it cannot change which query a
particular GT prefers; it only changes which queries the solver leaves unused.
Once every already-empty query has been pushed out, adding more weight only
distorts the ranking of the remaining queries.

The remaining untested shape is the **conditional** one, which is not
row-constant.

## Mechanism

Instead of adding a penalty, renormalise the class probability used by the
existing focal term over `{true class, no-object}`:

```
p_class  = sigmoid(scores)[:, gt_class]     # (nq, n_gt)
p_empty  = sigmoid(scores)[:, -1:]          # (nq, 1)
p_cond   = p_class / (p_class + p_empty + 1e-8)
cost_class = focal(p_cond)                  # existing alpha=0.25, gamma=2.0
```

Properties:

- A query that is confidently foreground (`p_class → 1`, `p_empty → 0`) keeps
  `p_cond → 1` and pays essentially nothing, same as today.
- A query with `p_class = p_empty = 0.5` drops to `p_cond = 0.5`, i.e. the focal
  term now sees a coin-flip foreground and charges for it. Today that query
  looks identical to a confident one as far as this GT is concerned.
- The term is **per (query, GT)**, because `p_class` depends on the GT's class
  column. Unlike the additive form it can reorder which query a GT prefers, not
  just which rows are wasted.
- It uses no geometry and no ray terms, so it cannot disturb `cost_xy` or
  `cost_radial`.
- `λ` is unused in this mode; the strength is fixed by the focal alpha/gamma and
  by how much probability mass the head actually puts in the empty column.

## Why this and not the alternatives

- Wrong-class distance is a guaranteed no-op: measured `p(correct) − p(best_wrong)`
  has median **0.000** on this head, so no cost built on class-vs-class margin
  can separate anything. The only signal-bearing pair is `{correct, empty}`,
  whose gap has median 0.269. This mechanism uses exactly that pair.
- Feeding `p_class − p_empty` directly is backwards (a confidently-empty query
  would look cheap). The ratio form is monotone in the right direction in both
  arguments.
- Allowing GTs to stay unmatched adds an unprincipled hyperparameter, scores an
  unassigned GT as 0 for that instance, and candidate supply is not the
  constraint (400 queries vs 8–44 GT per tile).

## Experiment

One run, one mechanism. `main/pannuke-thin-t12-empty-cost-cond.yaml` differs
from `main/pannuke-thin-t12-empty-cost.yaml` (the λ=1 additive arm) in exactly
one key:

```
detr_matcher_empty_cost_mode: additive -> conditional
```

and from the control in two keys (`detr_matcher_empty_cost: true`,
`detr_matcher_empty_cost_mode: conditional`). Verified with `ETLConfig`.

- `--epochs 600 --end 332`, batch 16, fold1 train / fold2 val / fold3 held out
- reuse `output/oracle/nuLiteT-fold1-sep-t12.pt` (seg stage skipped)
- ~6.6 h, same as every other T12 matcher arm

```
scripts/train_empty_cond.sh
# or, detached:
setsid nohup scripts/train_empty_cond.sh > output/logs/empty_cond_driver.log 2>&1 < /dev/null &
```

Run name: `train59_empty-cond`.

## Pre-registered decision rule

Evaluate with `raycasted/evals/eval_lsp.py` on the Fold3 parquet, raw `nc+1`
argmax, no threshold, no NMS (2,722 images / 66,654 GT).

- adopt only if **bPQ and macro bPQ both** gain **> +0.002** over the control
  (0.6372 / 0.6566);
- reject if bPQ drops **> 0.002**;
- always report `good_suppressed_bg` and `unmatched_predictions` cohort shifts
  (384-image audit cache, same protocol as the previous arms).

Because the nearest arm is the λ=1 additive run, also compare against it
directly: the interesting outcome is not "does it beat the control" but "does
the non-row-constant shape move the designed cohort further than +118".

## Code surface

- `raycasted/model/nulite_detr_loss.py` — `RayCastHungarianMatcher.__init__`
  gains `empty_cost_mode='additive'` (validated against
  `additive|conditional`); the `pred_sel`/focal block renormalises before the
  focal term when `conditional`, and only applies the additive term when
  `additive`. `NuLiteDETRLoss` forwards the flag.
- `raycasted/model/nulite_detr_model.py` — `detr_matcher_empty_cost_mode`
  attribute, passed in both `init_criterion` branches via `getattr` so old
  checkpoints still load.
- `raycasted/model/train.py` — reads `detr_matcher_empty_cost_mode` from the
  training block.
- `raycasted/data/etl/utils/config.py` — schema key (pydantic is
  `extra_forbidden`, so an unlisted key fails validation).
- `tests/phase_10/test_matcher_empty_cost_conditional.py` — 7 tests: invalid
  mode raises; flag-off is bit-identical to legacy; conditional differs from
  additive; cost increases with `p_empty`; cost decreases as empty mass is
  removed at equal `p_class`; a higher-class-score query with competing empty
  mass loses to a lower-class-score query with none; the term is skipped when
  `no_object` is disabled.

## Out of scope

- Re-tuning focal alpha/gamma for the renormalised probability (would be a
  second mechanism in the same run).
- Applying the same renormalisation to the classification *loss*; this is
  matcher-only, so the inference contract (`nc+1` argmax, no NMS) is unchanged.
- Any geometry change — the inside gate is settled negative.
