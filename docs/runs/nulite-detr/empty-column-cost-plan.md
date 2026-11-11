# Empty-column matcher cost — plan (2026-11-11)

Single-mechanism experiment: one config key, one run, one fold. Control is the
existing T12 base `train52_fastvit-t12`, already evaluated on the held-out fold
with `eval_lsp.py`.

This is a diagnostic, not a paper claim. One fold, one seed.

## Motivation

Two measurements from the T12 checkpoint (`output/v2/train52_fastvit-t12`) fix
what is worth trying and what is not.

1. **Class-vs-class distance carries no signal.**
   `p(correct) - p(best_wrong)` has median **0.000** and ranges only
   `-0.007 .. 0.000` across 1,241 GTs. The head does not separate nucleus
   types. Any matcher cost built on the *wrong-class* distance is therefore a
   guaranteed no-op.
2. **The foreground-vs-empty distance does carry signal.**
   `p(correct) - p(empty)` has median **0.269** (min 0.113). This is the only
   classification distance with headroom, and it is exactly the decision the
   one-to-one head is currently getting wrong: 71.6% of GTs have a nearby query
   with `p(empty) > 0.5`, i.e. a well-placed query that says "no object" and
   still absorbs the match. On Train50 this is the `good_suppressed_bg` cohort.

So "add class distance" and "add empty-column cost" collapse into the same
lever, and only the empty-column form can do anything.

## Mechanism

Every query's match cost gains a term that depends only on its own empty logit:

```
p_empty      = sigmoid(score)[:, -1]                  # (nq,)
cost_class  += lambda * -log(1 - p_empty + 1e-8)      # broadcast over GT columns
```

Default `lambda = 1.0`, frozen before the run.

Why this shape and not `p_correct - p_empty`: subtracting would make a
confidently-empty query look *cheap*, which is backwards. The additive
`-log(1 - p_empty)` is a proper penalty — 0 when the query is confidently
foreground, growing to ~4.6 when it is confidently empty — and it leaves the
existing focal term unchanged, so absolute foreground confidence still matters.

Why this can work at all: the term is **constant across GT columns** for a given
query. Hungarian's objective is unchanged for any fixed set of assigned
queries, so it cannot reorder an already-chosen assignment. What it changes is
*which queries the solver leaves unused* — the solver will now prefer to leave
a confidently-empty query idle and spend a different query on the nucleus.
With 400 queries against 8–44 GT per tile, unused rows are plentiful, so this
lever has room to act.

Guards: the term is only applied when `no_object` is enabled (there must be an
empty column) and is skipped otherwise, so `has_no_object=False` is a no-op
rather than a silent misread of the class dimension. Flag off is byte-identical
to the current matcher.

## Experiment

| run | recipe | change vs control |
|---|---|---|
| control | `main/pannuke-thin-t12.yaml` | — (already evaluated) |
| empty-cost | `main/pannuke-thin-t12-empty-cost.yaml` | `detr_matcher_empty_cost: true` |

Verified by parsing both with `ETLConfig`: the training block differs in exactly
that one key. `lambda` stays at its 1.0 default, `detr_matcher_lsp: true` and
`detr_no_object: true` are unchanged.

Schedule matches the whole T12 series: `--epochs 600 --end 332`, batch 16,
~6.5 h on one 5070 Ti.

```
scripts/train_empty_cost.sh
```

## Decision rule (fixed in advance)

Evaluate `output/v2/train57_empty-cost/weights/last.pt` on the genuinely
held-out Fold3 with `raycasted/evals/eval_lsp.py` — original dense parquet,
raw `nc+1` argmax, no threshold, no NMS.

- **Adopt** if bPQ **and** macro bPQ both gain more than **+0.002** over the
  control (0.6372 bPQ / 0.6566 macro bPQ).
- **Reject** if bPQ falls more than 0.002.
- **Always report** the cohort breakdown (`good_suppressed_bg`,
  `no_good_polygon`, `wrong_type`, `unmatched_predictions`) from
  `audit_lsp_cohorts.py`. A flat bPQ with a large suppression drop is still
  informative — it tells us the empty decision moved but the polygons did not
  follow, which is what the soft inside gate did.

## Out of scope

- **Allowing GTs to stay unmatched.** Adds an unprincipled hyperparameter, and
  an unassigned GT scores zero for that instance, so it fails loudly. Query
  supply is not the constraint (400 vs 8–44 GT/tile).
- **A lambda sweep.** One value, fixed a priori. If `lambda=1.0` is flat, a
  sweep is the natural follow-up rather than something to pre-commit to.
- **Wrong-class separation.** Median margin is 0.000; that is a
  classification-loss/representation problem worth ~11% of GT
  (`wrong_type`), and no matcher change can fix it.
- **The dead `detr_cost_inside` key.** It is still ignored under
  `detr_matcher_lsp`; documenting it is a separate cleanup.
