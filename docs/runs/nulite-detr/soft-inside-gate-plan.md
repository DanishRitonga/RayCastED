# Soft inside gate for the Hungarian matcher — plan and calibration

Status: implemented and unit-tested; **not yet trained**.

## Motivation

The one-to-one Hungarian matcher rejects every query whose centroid falls
outside the GT polygon via a constant `9999.0` penalty
(`raycasted/model/nulite_detr_loss.py`, LSP branch). This is a hard veto: an
outside query is un-matchable regardless of how well it fits geometrically or
how confident it is.

The measured consequence is that GTs are forced onto less suitable inside
queries. On 1,234 held-out GTs, 41.8% have a nearby *outside* query whose
foreground score exceeds every nearby inside query. The trained head's
foreground recall of 0.650 against a linear probe's 0.912 on the same features
(`docs/diag/foreground-map-not-used-2026-11-01.md` §6) is consistent with the
decision rule, not the representation, being the constraint.

## Mechanism

Replace the constant veto with a distance-scaled penalty:

```
excess = clamp(dist / d_phi - 1, min=0)        # 0 inside the GT star
cost_inside = min(w * excess**p, 9999)         # p = 2, w = 20
```

`d_phi` is the GT boundary radius interpolated between the two bracketing rays,
already computed by `_points_inside_star`. The new helper
`_points_inside_star_excess` returns both the boolean mask and the excess, so
`_points_inside_star` keeps its old signature.

Behaviour:

- Inside pairs pay exactly 0 at any weight — the inside population is unchanged.
- A barely-outside query pays a small finite amount and can compete.
- The far tail clamps at the legacy 9999, which keeps the matrix finite under
  fp16 AMP and guards degenerate zero-length GT rays. Any pair at the clamp could
  never have won anyway.

The clamps and weight are config-only: `detr_matcher_soft_inside`,
`detr_matcher_soft_inside_w`, `detr_matcher_soft_inside_p`. The flag defaults to
false and the off-path is byte-identical to the previous line.

## Calibration (frozen before any eval)

`w = 20`, `p = 2`, chosen from the measured excess distribution and then frozen.
Rationale: the outside-pair excess has median 0.495 and p90 0.848, so the
penalty is ~4.9 at the median (comparable to a typical `cost_xy` of 0–1.9) and
~14.4 at p90 (dominant). A boundary-adjacent pair (`excess` 0.04) pays 0.03.

**The calibration was deliberately not re-tuned after probing.** Every probe
runs a checkpoint trained *under* the hard gate, which never learned to place
well-shaped queries just outside. Its preference for the hard-gate assignment is
therefore circular. Only training can settle the weight.

## Probe evidence (read as diagnostic, biased as noted)

On `output/v2/train52_fastvit-t12/weights/last.pt`, 16 held-out Fold3 tiles:

| quantity | value |
|---|---|
| GT slots whose matched query changes | 27% (112/408 at w=20) |
| outside pairs strictly cheaper under soft | 76.7% of hard-gated pairs |
| flip rate, w = 2 / 20 / 200 / 1000 | 28.6% / 26.6% / 25.6% / 25.4% |
| total non-gate cost, w = 2 | −63 vs hard gate (cheaper) |
| total non-gate cost, w = 20 | +56 vs hard gate (costlier) |

Flip rate is nearly independent of `w`: most reassignments come from breaking
the degenerate ties the constant penalty created, not from the pair-level cost.

## Experiment

| run | recipe | change |
|---|---|---|
| control | `main/pannuke-thin-t12.yaml` | T12 base, already evaluated |
| soft-inside | `main/pannuke-thin-t12-soft-inside.yaml` | `detr_matcher_soft_inside: true` |

Exactly one key differs (verified through `ETLConfig`). `--epochs 600 --end 332`,
single fold. ~6.5 h.

```bash
scripts/train_soft_inside.sh          # trains train56_soft-inside only
```

This is a single-mechanism test, so it has its own one-run script rather than
riding the T12 grid runner. `scripts/train_ablation_t12.sh` keeps its six grid
variants and is unrelated to this experiment.

## Pre-registered decision rule

Evaluate with `eval_lsp.py`, raw `nc+1` argmax, no NMS, held-out Fold3.

- **Adopt** if bPQ *and* macro bPQ both gain more than +0.002 over the T12 base
  (0.6372 bPQ, 0.6566 macro bPQ).
- **Reject** if bPQ drops more than 0.002.
- **Always report** the `good_suppressed_bg` and `unmatched_predictions` cohort
  shifts. This change targets suppression specifically, so a flat bPQ with a
  large cohort shift is informative rather than a failure.

## Out of scope

- Classification distance in the matcher (`p(correct) − p(empty)` has a usable
  median margin of 0.269; `p(correct) − best_wrong` is 0.000, so the head does
  not separate nucleus types at all).
- Allowing GTs to stay unmatched.
- The dead `detr_cost_inside` key: under `detr_matcher_lsp: true` the matcher
  takes the LSP branch, which ignores `cost_gain` entirely.
