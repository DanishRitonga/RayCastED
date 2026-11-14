# Foreground/seed-map ablation sweep — results (2026-11-13, consumer 4 added 2026-11-14)

Four single-mechanism runs against `main/pannuke-thin-t12.yaml`
(`train52_fastvit-t12`). Each recipe is a copy of the control with exactly the
listed diff, same 600-epoch schedule stopped at 332, same fold1 → fold2 →
held-out fold3 split. The consumer-4 arm additionally required a one-key opt-in
(`detr_seed_map_loss: true`) because the supervision path is dead code by
default in the frozen layout.

Diagnostic, not a paper claim. One fold, one seed.

## Scope of the current manuscript — read this first

**For this version of the manuscript, ONLY the fg−bg (binary plateau) seed map
is used. The centroid-potential map is UNUSED for now.**

- Every checkpoint behind a manuscript number — Train50 (T8 baseline), the
  backbone sweep (T12 / S12 / SA12), the edgeNeXt variants, the T8 decoder grid,
  the T12 decoder grid, and all four matcher arms — was trained with
  `nuLiteT-fold1-sep-*.pt` and no `nulite_oracle_kind`, i.e. `oracle_kind:
  binary`, i.e. seed = `logit(fg) − logit(bg)`. Verified from the checkpoint
  attributes themselves.
- The centroid-potential oracles (`nuLiteT-fold1-center-t12.pt`,
  `nuLiteT-fold1-fg-center-t12.pt`) were built on 2026-10-02 / 2026-10-05 and
  no recipe in git history referenced them before 2026-11-13.
- Therefore the `fgcentroid` metrics in this document are **not adopted and are
  not part of any manuscript number**. They are recorded because this is the
  only test of a different map target.

Why the arm is still worth reading: it is below the adoption bar on PQ
(bPQ +0.0015, macro bPQ +0.0018, mPQ −0.0019) but it improves the detection
operating point (F1 +0.0047, precision +0.0043, recall +0.0049) and removes 36
`good_suppressed_bg` nuclei. That is a **directional hint, not evidence**, that
a properly calibrated centroid map could improve performance — most plausibly
the decision/recall side rather than segmentation quality. The proper test was
not run: the arm changed two things at once (map target *and* oracle
initialisation) and used a sharp σ = 2.5–4 px centroid channel that is nearly
dead at the 12.8 px query grid. Revisit with the wide-sigma oracle and the
oracle-init confound removed before claiming anything.

## Arm definitions

| arm | run dir | diff vs control | question |
|---|---|---|---|
| control | `output/v2/train52_fastvit-t12` | — | baseline |
| **nocontent** | `output/v2/train60_fgmap-nocontent` | `detr_seed_in_content: True → False` | does the map drive the fg/empty decision at all? |
| **noseedlevel** | `output/v2/train60_fgmap-noseedlevel` | `detr_seed_level: True → False` | is the 4th attention level load-bearing? |
| **fgcentroid** | `output/v2/train60_fgmap-fgcentroid` | `nulite_ckpt → nuLiteT-fold1-fg-center-t12.pt` **and** `nulite_oracle_kind: binary → fg_centroid` | does a clean, validated two-channel map beat the plateau? |
| **seedloss** | `output/v2/train61_fgmap-seedloss` | `detr_seed_map_loss: False → True` (opt-in; reuses the recipe's existing `lambda_seg: 1.0`) | does direct supervision of the seed map help? |

`fgcentroid` changes two keys but is one mechanism: `nulite_detr_model.py`
raises if the checkpoint's `oracle_kind` disagrees with the model's, so the
checkpoint and the kind cannot be separated.

Two semantic consequences of that swap, both part of the same change:

- the 4th attention level receives the **sharp centroid channel** (0.059 median
  at grid cell centres) instead of the broad plateau (0.882);
- the decision-path seed feature becomes a **single foreground logit**
  (`seed = ch0`) instead of the `logit(fg) − logit(bg)` difference, so the
  `nocontent` null result does not transfer literally to this arm.

The map has four consumers, and only one touches the decision:

| # | knob | path | decision? |
|---|---|---|---|
| 1 | `detr_seed_in_content` | seed logit concatenated to the score head (`score_in = hd + 1`, sampled at the query) | **yes — only this one** |
| 2 | `detr_seed_level` | `centroid_seed` projected as a 4th deformable-attention level | no (geometry) |
| 3 | `detr_hybrid_peaks` | local maxima of the seed map become queries — **inert here**: only read when `detr_query_selection: hybrid`, and every recipe sets `grid` | no (proposals) |
| 4 | `seed_map_target` + `lambda_seg` | L1 to the InstanSeg EDT target built from GT in collate | no (supervision) — **inert by default**: `lambda_seg` forced to 0 in the frozen branch; tested only through the `detr_seed_map_loss` opt-in |

Consumers 3 and 4 are therefore dead configuration in every shipped T12/T8
recipe, not merely untested knobs: `detr_hybrid_peaks` never executes under
`query_selection: grid`, and the EDT target is built in the collate fn and then
never used because `lambda_seg` is 0. Consumer 4 was made testable for the
`seedloss` arm with a minimal opt-in (`detr_seed_map_loss`, default `False`,
which leaves training bit-identical). All four consumers are now either tested
(1, 2, 4) or structurally inapplicable (3).

## Runs

All four arms completed 332 / 332 epochs.

| arm | wall time | s/epoch | final validator bPQ (trajectory only) |
|---|---:|---:|---:|
| control | 23,596.6 s = 6.55 h | 71.1 | 0.66005 |
| nocontent | 23,617.6 s = 6.56 h | 70.9 | 0.66118 |
| noseedlevel | 22,481.5 s = **6.24 h** | 67.5 | **0.66222** |
| fgcentroid | 23,647.3 s = 6.57 h | 71.0 | 0.65821 |
| seedloss | 23,691.9 s = 6.58 h | 71.4 | 0.65991 |

`noseedlevel` is 4.7 % faster, as expected from dropping one attention level;
the cost saving is real and unrelated to quality.

Validator order (noseedlevel > nocontent > control > fgcentroid) does **not**
survive held-out evaluation. It is recorded here only to make that point again:
the validator trajectory is not a result.

## Held-out Fold3

`eval_lsp`, 2,722 images, 66,654 GT, raw `nc+1` argmax, no confidence
threshold, no NMS, `weights/last.pt`. The control was re-evaluated in the same
session and reproduced its recorded numbers exactly.

| metric | control | nocontent | delta | noseedlevel | delta | fgcentroid | delta | seedloss | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bPQ | 0.6372 | 0.6365 | −0.0007 | 0.6332 | **−0.0040** | 0.6387 | +0.0015 | 0.6343 | **−0.0029** |
| bDQ | 0.7798 | 0.7780 | −0.0018 | 0.7755 | −0.0043 | 0.7811 | +0.0013 | 0.7754 | −0.0044 |
| bSQ | 0.8032 | 0.8052 | +0.0020 | 0.8027 | −0.0005 | 0.8029 | −0.0003 | 0.8031 | −0.0001 |
| mPQ | 0.4561 | 0.4561 | 0.0000 | 0.4504 | −0.0057 | 0.4542 | −0.0019 | 0.4519 | −0.0042 |
| macro bPQ | 0.6566 | 0.6568 | +0.0002 | 0.6523 | **−0.0043** | 0.6584 | +0.0018 | 0.6558 | −0.0008 |
| macro mPQ | 0.4618 | 0.4577 | −0.0041 | 0.4538 | −0.0080 | 0.4603 | −0.0015 | 0.4569 | −0.0049 |
| F1 (centroid r=12) | 0.7987 | 0.7999 | +0.0012 | 0.7969 | −0.0018 | **0.8034** | +0.0047 | 0.7964 | −0.0023 |
| precision | 0.8270 | 0.8265 | −0.0005 | 0.8299 | +0.0029 | **0.8313** | +0.0043 | **0.8395** | +0.0125 |
| recall | 0.7724 | 0.7749 | +0.0025 | 0.7664 | −0.0060 | **0.7773** | +0.0049 | 0.7574 | −0.0150 |

`fgcentroid` is the only arm that improves the detection-side operating point
(F1 / precision / recall) but it pays for that in `mPQ` (−0.0019) and
`macro mPQ` (−0.0015), and its bPQ gain (+0.0015) is under the adoption bar.

`seedloss` is the opposite trade, and the only arm that raises precision
(+0.0125). It does so by emitting 254 fewer raw queries, at the cost of recall
(−0.0150), `mPQ` (−0.0042) and bPQ (−0.0029). The mechanism is visible in the
loss: L1 on the seed map compares *logits* to a 0–1 EDT target, so the term is
≈ 2.91 at initialisation and the anchor pulls the map toward a flat,
mid-probability field. At `lambda_seg = 1.0` it is only ≈ 0.7 % of the total
loss, but it acts on 334 thawed U-Net parameters from epoch 50 on and makes the
head less willing to commit to a query — the designed cohort moved the wrong
way (`good_suppressed_bg` +115).

Per-tissue (fgcentroid, macro over 19 tissues): best Liver bPQ 0.7050,
worst Colon 0.5518 then Skin 0.6060 (Colon is also among the weakest tissues
for the control). Full table in `output/evals/fgmap/fgcentroid.log`.

## Cohort audit (384 images, 8,917 GT)

Counts read from `*_cohort.json` (`counts`).

| cohort | control | nocontent | delta | noseedlevel | delta | fgcentroid | delta | seedloss | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched | 6,929 | 6,917 | −12 | 6,845 | **−84** | 6,951 | **+22** | 6,808 | **−121** |
| good_suppressed_bg | 1,163 | 1,184 | +21 | 1,217 | **+54** | 1,127 | **−36** | 1,278 | **+115** |
| no_good_polygon | 798 | 787 | −11 | 820 | +22 | 799 | +1 | 806 | +8 |
| wrong_type | 974 | 994 | +20 | 999 | +25 | 995 | +21 | 972 | −2 |
| unmatched_predictions | 1,262 | 1,314 | +52 | 1,266 | +4 | 1,310 | +48 | 1,132 | **−130** |
| good_fg_unmatched | 27 | 29 | +2 | 35 | +8 | 40 | +13 | 25 | −2 |
| raw predictions | 8,196 | 8,234 | +38 | 8,112 | −84 | 8,261 | +65 | 7,942 | **−254** |

The `noseedlevel` loss is a clean decision-side signature: 84 fewer matches and
54 more well-formed polygons whose query stayed empty, with no compensating
gain elsewhere. Removing the 4th level did not just make polygons worse; it
made the head less willing to commit.

`fgcentroid` moves the decision cohort the right way (−36 suppressed, +22
matched) but emits 65 more raw queries and does not reduce `unmatched` — the
same precision-for-volume trade that capped the matcher arms.

`seedloss` is the only arm whose decision cohort moves in the *wrong*
direction: 121 fewer matches, 115 more well-formed-but-suppressed nuclei, and
130 fewer unmatched predictions. That combination is the signature of a head
that stops emitting rather than a head that emits badly: `no_good_polygon`
changes by only +8 and `raw predictions` falls by 254.

## Verdict against the pre-registered rules

Rules 1–4 were fixed in `fg-map-ablation-plan.md` before the runs; rule 5
carries the same pre-registered bar to the consumer-4 arm and was stated in
its runner before training.

1. **`nocontent` within ±0.002 on bPQ and macro bPQ → the map is not
   load-bearing for the fg/empty decision.** bPQ −0.0007, macro bPQ +0.0002.
   **Rule fires.** Deleting the only map path into the score head costs nothing
   measurable; the map-quality line aimed at the decision is closed.
2. **`nocontent` drop > 0.002 → the map is load-bearing.** Did not fire.
3. **`noseedlevel` drop > 0.002 → the 4th level matters.** bPQ −0.0040, macro
   bPQ −0.0043. **Rule fires.** The 4th deformable level is load-bearing, and
   its contribution is geometric/decision-side, not map-quality per se.
4. **`fgcentroid` / wide adopted only if bPQ AND macro bPQ both > +0.002.**
   bPQ +0.0015, macro bPQ +0.0018 → **not adopted**. Reject line (bPQ < −0.002)
   not approached.
5. **`seedloss` adopted only if bPQ AND macro bPQ both > +0.002; rejected if
   bPQ < −0.002.** bPQ −0.0029, macro bPQ −0.0008 → **rejected** (bPQ past the
   reject line). This is the same bar used for the four matcher arms; it was
   fixed before the run and restated in the arm's runner
   (`scripts/train_fgmap_seedloss.sh`).

No arm clears the adoption bar, so the control remains the baseline. The
`fgcentroid-wide` arm was not run: rule 1 already showed the decision path is
not where the map matters, so changing centroid kernel width cannot be expected
to move bPQ.

## What this establishes

- The seed map's influence on the foreground/empty decision is, on this
  architecture and fold, **below the measurement resolution** (−0.0007 bPQ,
  i.e. 1/6 of the pre-registered noise band). The diagnostic claim in
  `docs/diag/foreground-map-not-used-2026-11-01.md` — the map is informative at
  the exact centroid, noise at the grid cell centre, and the head over-weights
  the seed column — now has a **training-side** confirmation, not only an
  inference probe.
- The map's real contribution is **geometric**: the 4th deformable level is
  worth ~0.004 bPQ, delivered mostly through whether the head commits to a
  nucleus (matched −84, suppressed +54), not through polygon shape (bSQ −0.0005).
- A cleaner centroid target (`fg_centroid`, fg_dice 0.8103 / center_f1 0.7487)
  buys a better operating point (F1 +0.0047, P +0.0043, R +0.0049) and 36 fewer
  suppressed nuclei, but not PQ: mPQ −0.0019, macro mPQ −0.0015. It shifts the
  decision toward recall without improving segmentation quality where the
  matched set is concerned.
- **Making the map better by supervising it directly does not help either.**
  With the direct EDT anchor live (`seedloss`), the head emits 254 fewer
  queries, precision rises +0.0125 and PQ falls (bPQ −0.0029, mPQ −0.0042).
  Together with the `nocontent` null this says the map's *quality* is not the
  binding constraint in either direction: neither removing its decision
  contribution nor explicitly training its target improves PQ. What the map
  supplies is spatial support (consumer 2), and that is already adequate.

## What this does not establish

- Only one fold and one seed per arm. Deltas of ±0.004 at the arm level are at
  the edge of what a single seed can resolve; the `nocontent` ≈ 0 result is the
  most robust because it is a *null* with a tight bound, but the `noseedlevel`
  and `fgcentroid` magnitudes need the 3-fold rotation before being quoted as
  numbers.
- `fgcentroid` is confounded by the oracle swap: the map target changes *and*
  the thawed-after-epoch-50 oracle init changes. The arm measures "a different
  map kind", not "the decision path with a better map".
- `seed_map_target` + `lambda_seg` (consumer 4) is **inert by default**, not
  merely untested: the frozen/two-stream branch hard-sets
  `self.lambda_seg = 0.0` (`nulite_detr_model.py:367` and `:418`), and the seg
  loss block is guarded by `if seed is not None and self.lambda_seg > 0`
  (`:1351`). The InstanSeg EDT target is therefore built in the collate fn and
  never used, so `seed_map_target: true` in these recipes has no effect on the
  loss. The `seedloss` arm tests it through a one-key opt-in
  (`detr_seed_map_loss`, default `False`, bit-identical when absent). Two
  limits remain on how far that arm generalises:
  - it is a no-op for epochs 0–49 (the oracle is frozen, so the anchor has no
    gradient) and only acts on the 282 epochs after the thaw;
  - it was run at a single weight (`lambda_seg: 1.0`, ≈ 0.7 % of the total loss).
    A stronger anchor would be a different experiment and is untested.
- No single weight was swept, so this is "the anchor at recipe weight does not
  help", not "supervising the map cannot help".
- No arm addresses `wrong_type` (974 / 8,917 GT). That remains a class
  separation problem, not a matcher or map problem.

## Implications for the strategy

**Keep the control as the baseline.** Nothing in this sweep clears the bar, and
the one arm that improves F1/precision/recall (`fgcentroid`) is below the bar on
both PQ aggregates and requires a replacement oracle checkpoint — extra
machinery for a result inside the noise band.

**Manuscript stance on the map target (explicit).** This version of the
manuscript uses only the fg−bg plateau map. The centroid-potential map is not
used, `fgcentroid` is not adopted, and no centroid arm appears in the results
table of record. The arm stays in this document as a **future-work pointer**:
its direction suggests that a proper centroid map may improve the detection
operating point, but it is below the adoption bar, single-fold, and confounded,
so it cannot be presented as a result.

**For the manuscript**, the honest framing is narrower than "the effect is
minimal":

- Report this as an ablation, not a headline. State the mechanism: the seed map
  enters the head through exactly one decision path (`seed_in_content`) and
  three non-decision paths (4th attention level, peak proposals, EDT
  supervision).
- The claim worth making is directional and mechanistic: *the foreground map's
  contribution to the NMS-free foreground/empty decision is not measurable in
  this architecture; its contribution is geometric.* That directly supports the
  paper's framing that retain/empty is decided by the query head, not by
  thresholding a dense map — which is the entire point of the NMS-free
  contract.
- Do **not** write "the foreground map does not matter". `noseedlevel` shows it
  does matter (bPQ −0.0040, and it is the only arm that moves the
  suppressed/matched balance the wrong way). The safe sentence is: the map's
  role is to supply spatial support to the decoder, not a per-query
  foreground score.
- Do **not** quote the single-fold numbers in the abstract or results table of
  record. They are diagnostic. Run the 3-fold rotation on the chosen baseline
  and report mean ± sd across the three held-out folds; the ablation can then be
  reported once, on one fold, explicitly labelled as an ablation of the
  mechanism.

**What is now ruled out** (so the paper does not need to try them): matcher cost
reshaping (four arms, all inside ±0.002 or negative), decoder depth/width/layer
sharing (ten configs across two backbones, spread 0.0032), map-quality work
aimed at the decision, and direct seed-map supervision at the recipe weight
(one arm, bPQ −0.0029). The remaining unexplained loss is `wrong_type` (class
separation, `p(correct) − p(best_wrong)` median 0.000) and the residual
`good_suppressed_bg` population (~1,127 nuclei even in the best arm).

## Manuscript-ready summary of the foreground-map ablations

Written to be lifted into the paper's ablation subsection. All numbers are
held-out Fold3, `eval_lsp`, raw `nc+1` argmax, no threshold, no NMS, `last.pt`,
control `output/v2/train52_fastvit-t12` (bPQ 0.6372, macro bPQ 0.6566, mPQ
0.4561, macro mPQ 0.4618, F1 0.7987, P 0.8270, R 0.7724). Δ is vs that control.
Every arm changes training only; the inference contract is identical.

### What was varied

Every cell is `absolute (Δ vs control)`. Raw values are given so the table can
be used directly; the Δ is only a convenience.

| arm | change | bPQ | bDQ | bSQ | mPQ | macro bPQ | macro mPQ | F1 | P | R | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | fg−bg plateau map, all consumers at recipe values | 0.6372 | 0.7798 | 0.8032 | 0.4561 | 0.6566 | 0.4618 | 0.7987 | 0.8270 | 0.7724 | baseline |
| nocontent | map removed from the per-query score | 0.6365 (−0.0007) | 0.7780 (−0.0018) | 0.8052 (+0.0020) | 0.4561 (0.0000) | 0.6568 (+0.0002) | 0.4577 (−0.0041) | 0.7999 (+0.0012) | 0.8265 (−0.0005) | 0.7749 (+0.0025) | null |
| noseedlevel | map-carrying 4th attention level removed | 0.6332 (**−0.0040**) | 0.7755 (−0.0043) | 0.8027 (−0.0005) | 0.4504 (−0.0057) | 0.6523 (**−0.0043**) | 0.4538 (−0.0080) | 0.7969 (−0.0018) | 0.8299 (+0.0029) | 0.7664 (−0.0060) | **hurts** |
| fgcentroid | plateau map → validated two-channel fg+centroid map | 0.6387 (+0.0015) | 0.7811 (+0.0013) | 0.8029 (−0.0003) | 0.4542 (−0.0019) | 0.6584 (+0.0018) | 0.4603 (−0.0015) | **0.8034 (+0.0047)** | 0.8313 (+0.0043) | 0.7773 (+0.0049) | below bar |
| seedloss | direct EDT supervision of the seed map switched on | 0.6343 (**−0.0029**) | 0.7754 (−0.0044) | 0.8031 (−0.0001) | 0.4519 (−0.0042) | 0.6558 (−0.0008) | 0.4569 (−0.0049) | 0.7964 (−0.0023) | **0.8395 (+0.0125)** | 0.7574 (−0.0150) | **hurts** |

All arms: 332/332 epochs, identical schedule, identical fold split; only the
listed mechanism changes. `noseedlevel` is also 4.7 % faster (one fewer
attention level).

Decision-side cohort counts (384 images, 8,917 GT; absolute, with Δ in
brackets) that should accompany any of these claims:

| cohort | control | nocontent | noseedlevel | fgcentroid | seedloss |
|---|---:|---:|---:|---:|---:|
| matched | 6,929 | 6,917 (−12) | 6,845 (−84) | 6,951 (+22) | 6,808 (−121) |
| good_suppressed_bg | 1,163 | 1,184 (+21) | 1,217 (+54) | 1,127 (−36) | 1,278 (+115) |
| no_good_polygon | 798 | 787 (−11) | 820 (+22) | 799 (+1) | 806 (+8) |
| wrong_type | 974 | 994 (+20) | 999 (+25) | 995 (+21) | 972 (−2) |
| unmatched_predictions | 1,262 | 1,314 (+52) | 1,266 (+4) | 1,310 (+48) | 1,132 (−130) |
| raw predictions | 8,196 | 8,234 (+38) | 8,112 (−84) | 8,261 (+65) | 7,942 (−254) |

### Claims that are safe to write

1. *The map does not carry the foreground/empty decision.* "Removing the seed
   map from the per-query foreground score changes held-out PQ by −0.0007 bPQ
   and +0.0002 macro bPQ — below the ±0.002 resolution of a single-fold
   comparison. The retain/empty decision is therefore carried by the query
   head, not by thresholding a dense map."
2. *Its measurable contribution is spatial.* "The map's load-bearing path is
   the fourth deformable-attention level: removing it costs 0.0040 bPQ and
   0.0080 macro mPQ, mostly by leaving well-formed nuclei unclaimed (84 fewer
   matches, 54 more suppressed) rather than by degrading polygon shape
   (bSQ −0.0005)."
3. *Supervising the map does not recover that.* "Adding direct
   distance-transform supervision to the thawed segmentation U-Net lowers PQ
   (bPQ −0.0029) while raising precision (+0.0125) and cutting 254 raw queries:
   the anchor moves the operating point, not the representation."
4. *A better map is a future-work pointer, not a result.* "A validated
   centroid-potential map (fg-Dice 0.810, centre-F1 0.749) shifted the operating
   point toward recall (F1 +0.0047, recall +0.0049) without improving PQ
   (mPQ −0.0019); we do not adopt it."

### Claims that are not safe

- **"The foreground map does not matter."** `noseedlevel` contradicts it
  (bPQ −0.0040) and is the only arm that moves the suppressed/matched balance
  the wrong way.
- **"A better map improves PQ."** `fgcentroid` is +0.0015 bPQ, below the bar,
  single-fold, and confounded (map target and oracle initialisation change
  together).
- **"Supervision weight was tuned."** Only `lambda_seg: 1.0` was run; no weight
  sweep exists, so the seedloss result is "this weight does not help", not
  "supervision cannot help".

### Mechanics to state in the paper

Four consumers exist; only one touches the decision. The seed map is consumed by
(1) the per-query score (`seed_in_content`, the only decision path), (2) a
fourth deformable-attention level, (3) peak-derived queries, and (4) direct
segmentation supervision. Consumers (3) and (4) are inert in the shipped
recipes — peak queries are not selected (`query_selection: grid`) and the
segmentation loss weight is zeroed in the frozen layout — so the ablation
cleanly separates the decision path (1) from the spatial path (2). The
`seedloss` arm exists only to test (4) explicitly.

### Reporting requirements

- All arms are single-fold, single-seed diagnostics. Report them in the
  ablation subsection, explicitly labelled, never in the results table of
  record; run the 3-fold rotation on the final baseline first.
- Scope statement for this manuscript version: only the fg−bg plateau map is
  used; the centroid-potential map is not used and no centroid arm enters the
  results table.
- The ablation supports the paper's NMS-free framing: retain/empty is decided
  by the query head, and the dense map supplies spatial support rather than
  per-query confidence.

## Artefacts

- `output/v2/train60_fgmap-{nocontent,noseedlevel,fgcentroid}/` and
  `output/v2/train61_fgmap-seedloss/` — run dirs, `results.csv`,
  `weights/last.pt` (seedloss: 332/332 ep, 23,691.9 s, validator bPQ 0.65991)
- `output/evals/fgmap/` and `output/evals/fgmap_seedloss/` — `run.sh`,
  `driver.log`, per-arm `{log,npz,audit.log,_cohort.json,_cohort.csv}`, all
  exits 0
- recipes: `main/pannuke-thin-t12-fgmap-{nocontent,noseedlevel,fgcentroid,fgcentroid-wide,seedloss}.yaml`
- scripts: `scripts/train_fgmap_sweep.sh`, `scripts/train_oracle_fgcenter_wide.sh`,
  `scripts/train_fgmap_seedloss.sh`
- code hook for consumer 4: `detr_seed_map_loss` (schema key, default `False`;
  `nulite_detr_model.py` `seed_map_loss` flag), tests
  `tests/phase_10/test_seed_map_loss.py`

## Out of scope

- `fgcentroid-wide` (σ 4–8 px oracle) — not trained; rule 1 makes it unmotivated.
- `detr_hybrid_peaks` count sweep (consumer 3) — not applicable: the path only
  runs under `detr_query_selection: hybrid`, and every recipe sets `grid`
  (`num_queries` is 400 at load, confirming `_hybrid_decoder_input` never
  executes). Enabling it on the current binary map is also unpromising on its
  own: measured on the control over 64 Fold3 tiles, a 7×7 local-max pass
  (`peak_distance: 3`) yields 13.6 peaks/image for 30.2 nuclei and only 2.9% of
  GT have a peak within 3 px — the plateau's maxima are tie-break artifacts,
  not nucleus centres. This is one more reason a *centroid* map is the
  prerequisite for peak-derived queries.
- `lambda_seg` weight sweep for consumer 4 — only `1.0` was run (≈ 0.7 % of the
  total loss, active from the epoch-50 thaw). The arm is a single point, not a
  curve; a stronger anchor is a different, untested experiment.
- Consumer 3 on a centroid map — the only variant of the peak-query path worth
  testing, and it needs the centroid oracle that this manuscript version does
  not use.
- 3-fold rotation (required before any literature claim, per AGENTS.md).
