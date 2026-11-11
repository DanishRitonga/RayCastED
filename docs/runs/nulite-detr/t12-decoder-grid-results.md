# T12 decoder grid — results (2026-11-10)

Companion to `t12-decoder-grid-plan.md`, which fixed the decision rules before
launch. Six T12 decoder variants, single fold, single seed, trained
`--epochs 600 --end 332` exactly like the T8 grid and the Train30–50 series.

**Read this as a diagnostic, not a paper claim.** One fold, one seed, and every
variant lands inside the pre-registered ±0.002 noise band. The 3-fold rotation
does not exist yet.

## Provenance

- Evaluator: `raycasted.evals.eval_lsp` — the sole literature-facing evaluator.
- Held-out Fold3, original dense PanNuke parquet (2,722 images, 66,654 GT).
- Raw `nc+1` all-class argmax, **no confidence threshold, no NMS**, no
  post-processing. Overlap pixels resolved by greatest Euclidean interior
  distance (LSP-DETR test-time mask ownership).
- Checkpoints: `last.pt` from each completed run.
- Base reference is `train52_fastvit-t12` evaluated in the same session with the
  same command.

Note on the record: the first evaluation pass wrote its logs to `/tmp`, which is
tmpfs on this host, and the machine rebooted before the numbers were written
down. The table below was transcribed from that session and then **re-verified**
by re-running all seven evaluations into `output/evals/t12_grid/`. Every figure
reproduced to the last digit, as expected for a deterministic raw-argmax
protocol with no sampling or NMS.

## Runs

All six completed 332/332 epochs. Wall times are driver-reported cumulative
seconds divided by epoch count.

| run | recipe | s/epoch | wall |
|---|---|---:|---:|
| `train55_ablation-t12-ndl6` | `pannuke-thin-t12-ndl6.yaml` | 75.7 | 6.98 h |
| `train55_ablation-t12-ndl4` | `pannuke-thin-t12-ndl4.yaml` | 65.9 | 6.07 h |
| `train55_ablation-t12-g2` | `pannuke-thin-t12-g2.yaml` | 70.6 | 6.51 h |
| `train55_ablation-t12-g0` | `pannuke-thin-t12-g0.yaml` | 71.1 | 6.56 h |
| `train55_ablation-t12-hd128` | `pannuke-thin-t12-hd128.yaml` | 66.3 | 6.12 h |
| `train55_ablation-t12-hd256` | `pannuke-thin-t12-hd256.yaml` | 74.8 | 6.89 h |

Each recipe differs from `pannuke-thin-t12.yaml` in exactly one key, verified by
diffing the parsed `ETLConfig` training block.

## Held-out Fold3

| variant | params | bPQ | bDQ | bSQ | mPQ | macro bPQ | macro mPQ | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **T12 base** | 9.55M | **0.6372** | **0.7798** | 0.8032 | 0.4561 | **0.6566** | 0.4618 | 0.8270 | 0.7724 |
| ndl4 | 8.87M | 0.6369 | 0.7788 | 0.8045 | 0.4569 | 0.6556 | 0.4588 | 0.8262 | 0.7709 |
| ndl6 | 10.03M | 0.6362 | 0.7791 | 0.8033 | 0.4535 | 0.6568 | 0.4571 | 0.8271 | 0.7702 |
| g2 | 9.55M | 0.6363 | 0.7785 | 0.8044 | 0.4553 | 0.6554 | 0.4588 | 0.8265 | 0.7686 |
| g0 | 9.55M | 0.6355 | 0.7775 | 0.8022 | 0.4536 | 0.6553 | 0.4561 | **0.8347** | 0.7662 |
| hd128 | **8.10M** | 0.6340 | 0.7772 | 0.8032 | 0.4531 | 0.6542 | 0.4565 | 0.8180 | **0.7724** |
| hd256 | 11.37M | 0.6363 | 0.7776 | 0.8042 | **0.4574** | 0.6548 | **0.4621** | 0.8257 | 0.7709 |

Precision/recall are the centroid F1 block (radius 12, global).

## Verdict against the pre-registered rules

The plan required a config to move **both** whole-fold bPQ and macro bPQ up by
more than ~0.002 to count.

- **No variant qualifies.** The best bPQ is the base itself (0.6372). The
  best-to-worst spread across all six ablations is 0.0032, and five of six sit
  below the base.
- `g2` loses again (−0.0009 bPQ vs base, −0.0012 macro bPQ). Across T8 and T12
  this is now the one decoder knob that reproducibly hurts: 3 shared groups out
  of 5 steps is worse than 2 or 0.
- `hd128` is the *worst* accuracy at the *lowest* parameter count (0.6340 bPQ,
  −0.0032). Layer width is not free on T12 the way it looked on T8.
- `ndl6` and `hd256` buy +0.0012/+0.0013 macro mPQ but lose bPQ. Neither clears
  the threshold.

## The T8 result does not transfer

| knob | T8 held-out bPQ | T12 held-out bPQ | T8 read | T12 read |
|---|---:|---:|---|---|
| ndl6 | 0.6373 (+0.0036) | 0.6362 (−0.0010) | clear winner | no change |
| hd128 | 0.6333 (−0.0004) | 0.6340 (−0.0032) | thinning pick | regression |
| g2 | 0.6329 (−0.0008) | 0.6363 (−0.0009) | loses | loses |

The T8 conclusion that "hd128 is the thinning pick, ndl6 the accuracy pick" is
**backbone-specific and does not replicate**. Any document citing the T8 thinning
recommendation must state this.

## What this means

Decoder geometry — depth, hidden width, layer sharing — is exhausted as a lever.
Ten configs across two backbones produce a spread smaller than fold-to-fold
variation. This is consistent with the earlier finding that the binding
constraint is the one-to-one Hungarian decision rule, not representation or
capacity (`docs/diag/foreground-map-not-used-2026-11-01.md`, §6: a linear probe
on the same proposal features reaches AUC 0.912 while the trained head operates
at recall 0.650).

The productive next step is the matcher, not another decoder grid.

## Artefacts

- Re-verification logs: `output/evals/t12_grid/<variant>.log` (persistent).
- Run directories: `output/v2/train55_ablation-t12-<variant>/`.
- Recipe/runner: `main/pannuke-thin-t12-<variant>.yaml`,
  `scripts/train_ablation_t12.sh`.
