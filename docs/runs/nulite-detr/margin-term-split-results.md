# Margin-term split (seg_fg_margin vs viable_owner_fg) — results (2026-11-17)

Two single-mechanism removal runs against `main/pannuke-thin-t12.yaml`
(`train52_fastvit-t12`), splitting the joint `train63_nomargin` ablation into
its two terms. Diagnostic, not a paper claim. One fold, one seed.

## Why the split

`main/pannuke-thin-t12-nomargin.yaml` (`train63_nomargin`) turned both cohort
foreground-margin losses off at once and lost 0.0083 bPQ / 0.0059 macro bPQ,
with `good_suppressed_bg` 1,163 → 1,620 and matched −452. That arm cannot say
which term carries the effect, and the two are structurally different:

| term | weight | shape |
|---|---:|---|
| `L_seg_fg_margin` | 0.5 | **evidence-gated**: matched queries only, weight = 3×3 `grid_sample` of `sigmoid(seed)` at the predicted centroid (±4 px), hinge `ReLU(1 + z_empty − z_true)` |
| `L_viable_owner_fg` | 0.5 | **ungated**: every matched query whose detached sampled pIoU ≥ 0.5, `softplus(1 + z_empty − z_true)` |

Both are final-decoder-layer only and both are guarded by `> 0` in
`nulite_detr_model.py`, so `0.0` removes the term cleanly.

## Arms

| arm | run dir | diff vs control |
|---|---|---|
| control | `output/v2/train52_fastvit-t12` | — |
| segonly | `output/v2/train64_margin-segonly` | `detr_viable_owner_fg_weight: 0.5 → 0.0` |
| viableonly | `output/v2/train65_margin-viableonly` | `detr_seg_fg_margin_weight: 0.5 → 0.0` |
| nomargin | `output/v2/train63_nomargin` | both → 0.0 (2026-11-15) |

Each split recipe differs from the control in exactly one key (verified with
`ETLConfig`). All runs use `--epochs 600 --end 332`.

| run | epochs | wall time | s/epoch |
|---|---:|---:|---:|
| control | 332 | 23,596.6 s = 6.55 h | 71.1 |
| segonly | 332 | 21,768.8 s = 6.05 h | 65.6 |
| viableonly | 332 | 22,168.2 s = 6.16 h | 66.8 |
| nomargin | 332 | 20,306.1 s = 5.64 h | 58.9 |

Validator bPQ (trajectory signal only, not a result): control 0.66005,
segonly 0.65837, viableonly 0.65795, nomargin 0.65667. The validator ranks both
split arms slightly below control while the held-out evaluation puts one of
them marginally above it — another case of the validator not being a result.

## Held-out Fold3

`eval_lsp`, 2,722 images / 66,654 instances, raw `nc+1` argmax, no confidence
threshold, no NMS, `last.pt`. Control re-evaluated in the same session and
reproduced 0.6372 exactly.

| metric | control | segonly | Δ | viableonly | Δ | nomargin | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| bPQ | 0.6372 | 0.6373 | +0.0001 | 0.6354 | −0.0018 | 0.6289 | −0.0083 |
| bDQ | 0.7798 | 0.7773 | −0.0025 | 0.7750 | −0.0048 | 0.7647 | −0.0151 |
| bSQ | 0.8032 | 0.8056 | +0.0024 | 0.8057 | +0.0025 | 0.8068 | +0.0036 |
| mPQ | 0.4561 | 0.4566 | +0.0005 | 0.4530 | −0.0031 | 0.4524 | −0.0037 |
| macro bPQ | 0.6566 | 0.6574 | +0.0008 | 0.6562 | −0.0004 | 0.6507 | −0.0059 |
| macro mPQ | 0.4618 | 0.4628 | +0.0010 | 0.4574 | −0.0044 | 0.4576 | −0.0042 |
| F1 | 0.7987 | 0.7982 | −0.0005 | 0.7969 | −0.0018 | 0.7945 | −0.0042 |
| P | 0.8270 | 0.8425 | +0.0155 | 0.8408 | +0.0138 | 0.8756 | +0.0486 |
| R | 0.7724 | 0.7584 | −0.0140 | 0.7574 | −0.0150 | 0.7271 | −0.0453 |

## Cohort audit

384 images, 8,917 GT, counts read from `*_cohort.json`.

| cohort | control | segonly | viableonly | nomargin |
|---|---:|---:|---:|---:|
| matched | 6,929 | 6,785 (−144) | 6,787 (−142) | 6,477 (−452) |
| good_suppressed_bg | 1,163 | 1,331 (+168) | 1,293 (+130) | 1,620 (+457) |
| no_good_polygon | 798 | 781 (−17) | 812 (+14) | 808 (+10) |
| wrong_type | 974 | 966 (−8) | 975 (+1) | 916 (−58) |
| unmatched_predictions | 1,262 | 1,117 (−145) | 1,114 (−148) | 860 (−402) |
| good_fg_unmatched | 27 | 20 | 25 | 12 |
| raw predictions | 8,196 | 7,903 (−293) | 7,902 (−294) | 7,339 (−857) |

## Verdict against the pre-registered rule

Rule, fixed before the runs (both bPQ and macro bPQ): |Δ| ≤ 0.002 → that term
is not load-bearing; removal drops bPQ > 0.002 → load-bearing; removal gains
bPQ > 0.002 → drop it from the baseline.

- **`detr_viable_owner_fg_weight` is not load-bearing.** Removal is flat on
  bPQ (+0.0001) and macro bPQ (+0.0008), and `segonly` is slightly better than
  control on mPQ (+0.0005) and macro mPQ (+0.0010).
- **`detr_seg_fg_margin_weight` is not load-bearing either.** Removal costs
  0.0018 bPQ and 0.0004 macro bPQ, both inside the band.
- **The pair is load-bearing, super-additively.** Single removals sum to
  −0.0017 bPQ; removing both costs −0.0083, roughly 5× the sum. Counts agree:
  matched −144 and −142 individually versus −452 jointly;
  `good_suppressed_bg` +168 and +130 versus +457.

No term clears the bar for removal on its own, so nothing is adopted from this
sweep. The finding is structural, not a tuning result.

## What this establishes

- The model needs **at least one** class-vs-empty margin term. With neither
  present the head reverts to suppressing real nuclei (matched −452,
  `good_suppressed_bg` +457, raw predictions −857) and loses 0.0083 bPQ.
- The requirement is **implemented twice**: an evidence-gated seed-stencil
  hinge and an ungated `softplus` over viable owners. Either one alone
  sustains it, which is why each single removal is flat.
- Both single removals move the operating point the same way — precision
  +0.014, recall −0.014, F1 flat — the signature of two terms doing the same
  job rather than complementary jobs.
- The earlier `train63_nomargin` attribution ("the margin losses are strongly
  load-bearing, they account for ~457 suppressed nuclei") is correct only for
  the pair. Attributing the effect to a specific term was not supported and is
  now explicitly ruled out.

## What this does not establish

- Which term is preferable for a *simplified* baseline. `segonly` is better on
  every macro metric (macro bPQ +0.0008, macro mPQ +0.0010, mPQ +0.0005) at
  identical bPQ, so it is the natural single-term candidate, but that is a
  preference inside noise, not a measured gain.
- Whether the single-term configurations are stable under a different seed or
  fold. The redundancy could be seed-specific; only a 3-fold rotation settles
  it.
- Anything about the class axis. `wrong_type` is essentially unchanged by every
  arm here (974 → 966 / 975 / 916), consistent with the established fact that
  `p(correct) − p(best_wrong)` has median 0.000 on this head.

## Artefacts

- Recipes: `main/pannuke-thin-t12-margin-segonly.yaml`,
  `main/pannuke-thin-t12-margin-viableonly.yaml`
- Runner: `scripts/train_margin_split.sh`
- Runs: `output/v2/train64_margin-segonly`, `output/v2/train65_margin-viableonly`
  (joint arm `output/v2/train63_nomargin`)
- Evaluation: `output/evals/margin_split/` (`ctrl|segonly|viableonly` `.log`,
  `.npz`, `*_cohort.json/csv`, `*_audit.log`, `driver.log`); the joint arm's
  logs and cohort summaries are in `output/evals/recast_margin_eval/`
- CSV caveat: a `results.csv` column exists only while its weight is `> 0`, so
  the split arms have 32 columns and bPQ/bSQ/bDQ sit at fields 13/14/15
  (control: 33 columns, fields 14/15/16; nomargin: 29 columns, fields 12/13/14).
