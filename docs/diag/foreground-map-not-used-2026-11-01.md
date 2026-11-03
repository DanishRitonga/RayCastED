# Why the foreground map is not used (2026-11-01)

Read-only probes on frozen checkpoints. Companion to
`query-classification-failures-2026-10-28.md` (the suppression wall) and
`undocumented-findings-2026-10-30.md` (feature ceiling, oracle-off, sampling).

All numbers are held-out Fold3 unless stated. `t50` =
`output/v2/train50_thin-t8-aggressive/weights/last.pt` (6.16 M, thin FastViT-T8,
frozen NuLite-T8 oracle + fusion adapters + full thaw).

## The question

The model suppresses ~8.8 k nuclei that already have a usable polygon. The
oracle's foreground map is present (4th deformable level, plus a scalar in the
score head) yet the head effectively ignores it. Why?

## 1. The map's signal exists only at the exact nucleus centre

Sampling the raw seed map at two locations, same 4,395 GTs (200 tiles):

| sample location | median value | fraction > 0.5 |
|---|---:|---:|
| exact GT centroid | **0.768** | 73.4% |
| grid cell centre (what the model consumes) | — | — |

Discriminative power for matched-vs-suppressed at the **grid cell centre**:

| feature | AUC |
|---|---:|
| seed scalar | **0.503** |
| 3x3 local max | 0.515 |
| multi-scale patches (raw / avg3 / avg5 / avg9 / avg13) | 0.494-0.526 |
| seed + patch5 + local max | 0.544 |
| content (256-d, linear) | **0.722** |
| content + seed | 0.722 |

A grid query can sit up to half a cell (12.8 px) from the nucleus centre. At
that offset the scalar is statistically indistinguishable from noise, so the
head is right to discount it. `content + seed` equals `content` alone to three
decimals: the seed column contributes literally nothing.

## 2. The head is NOT under-weighting the map (salience hypothesis rejected)

If the seed were merely "1 of 257 equally important inputs", its first-layer
weight column would have norm ~1/sqrt(256) = 0.062 of a content row. Measured
on t50:

| head | seed column norm | content mean row norm | ratio | "fair share" |
|---|---:|---:|---:|---:|
| `enc_score_head` | 0.1455 | 1.1122 | **0.131** | 0.062 |
| `dec_score_head[0]` | 0.3618 | 1.6375 | **0.221** | 0.062 |

The map is weighted **2.1x-3.5x above a fair share**. The network knows the
input exists and treats it as more important than average; that weight still
buys no discrimination because the value is uninformative at the sampled
location. Giving an uninformative input more weight only amplifies noise.

The decoder head weights it more heavily than the encoder head (0.221 vs
0.131), consistent with per-layer re-sampling at the refined centroid
(`layer_seed_q` at `refined[..., :2]`): as the refined point approaches the true
centre the value becomes more informative, so the model learns to trust it
more. That is the behaviour predicted by a *location* problem, not a *salience*
problem.

## 3. Local maxima are the wrong operator for this target (partly)

The target is a **per-instance normalised distance transform**, not a centroid
map (`raycasted/model/train.py`, `_raycast_collate_fn`):

```python
dmax = float(dist[m].max())
seed[m] = dist[m] / dmax          # centre = 1 for EVERY nucleus, regardless of size
seed = (seed - 0.5) * 15.0        # -> [-7.5, +7.5]
```

Consequences: (a) amplitude carries no size/confidence information because every
instance is normalised to the same peak height; (b) `dist/dmax == 1` holds along
the entire medial axis, so elongated nuclei have *ridges*, not points, and a 3x3
local-max extractor returns several maxima per nucleus; (c) instance boundaries
come from polygon rasterisation while interiors come from an EDT of the union, so
the map is discontinuous across contacts — exactly where the excess peaks live.

Connected components are a much better inverse operator for a flat-top field
(193 tiles / 4,395 GT, t50):

| threshold | local maxima / img | CC / img | GT / img |
|---|---:|---:|---:|
| > 0.0 | 1,201 (52.8x) | **175 (7.7x)** | 22.8 |
| > 2.0 | 700 (30.8x) | **59 (2.6x)** | 22.8 |
| > 4.0 | 125 (5.5x) | 105 (4.6x) | 22.8 |
| > 6.0 | 0.45 | 0.45 | 22.8 |

CC extraction cuts spurious proposals ~12x at tau>2. **But coverage is the
binding limit:**

| threshold | GT with a CC covering it | CCs containing a GT |
|---|---:|---:|
| > 2.0 | **0.464** | 0.301 |
| > 4.0 | 0.123 | 0.277 |

At tau>2 more than half of all nuclei have no component over them at all, and
raising the threshold makes it far worse. There is no threshold where coverage
is high and precision is acceptable. A good centroid target would give ~1.0x
ratio and ~100% coverage here.

## 4. The oracle's job was never foreground detection

Zeroing the oracle at inference (train34, 24.7 M two-stream control) costs
bPQ 0.6325 -> 0.6044 (-0.0281), macro bPQ -0.0368, but the damage is
**geometric**, not decision-related:

| cohort | baseline | oracle zeroed | delta |
|---|---:|---:|---:|
| no_good_polygon | 6,654 | 7,541 | **+887** |
| matched | 50,112 | 48,896 | -1,216 |
| good_suppressed_bg | 9,764 | 10,106 | **+342** |
| wrong_type | 8,172 | 7,917 | -255 |

The map's real function is the 4th deformable attention level supplying spatial
evidence (bSQ -0.0160 when removed). It barely moves the suppression wall.

## 5. Every attempt to route the map into the decision was firewalled

- **nucleus-CDN** (`detr_nucleus_cdn`): DN queries received a private
  `denoising_class_embed` carrying the answer. Probe: swapping that row for a
  different class costs -7.88 margin on a trained model vs -0.29 on a fresh one;
  the DN token and seed evidence have exactly 0.000 effect. CDN loss fell fast
  while the real decision boundary was untouched (0.6302).
- **CDN token-only**: shortcut removed -> confirmatory null (0.6251).
- **real->DN attention open**: the model learned to depend on the training-only
  scaffold and collapsed at eval (recall -> 0.001 by ep20).
- **`DenseAuxHead`** (`aux_dense_weight`): output stashed in `_last_aux_dense`
  and never reaches `predict`. Decorative.
- **`seed_in_content`**: one scalar among 257, and one-to-one Hungarian never
  penalises ignoring it because a neighbouring query usually owns the nucleus.
- **`detr_map_modulation` + `seed_in_content_pooled`** (t47): the projection did
  grow (0.120 by ep87, vs exactly 0.0 at init) but the perturbation was ~1-2% of
  content norm and bPQ stayed inside +/-0.002 noise through ep133. Stopped.

## 6. The wall is a decision-rule problem, not a representation one

On the exact 257-d proposal input, matched vs `good_suppressed_bg`:

| probe | AUC | recall | tnr |
|---|---:|---:|---:|
| trained head | 0.8489 | 0.650 | 0.926 |
| balanced linear probe, same features | **0.9124** | **0.931** | 0.638 |
| MLP 512-512 | 0.8967 | 0.939 | 0.681 |

Size-matched subset (760 vs 760, median 178 vs 183 px, log-area AUC 0.5056 =
chance): linear 0.8918, MLP 0.8914. More capacity does not help (MLP <= linear).

The information needed is already present. The trained head sits at recall 0.650
because one-to-one Hungarian over ~400 queries with ~93% negatives rewards
precision on the majority matched class.

## 7. Where the matcher lets this through

`RayCastHungarianMatcher.forward` (LSP mode, the active path):

```
cost = cost_class + cost_xy + cost_radial + 9999 * (~inside)
pred_sel = p_scores[:, g_cls]          # true class only; empty column ABSENT
```

Two structural gaps:

1. **The empty column never participates in matching.** Assignment compares
   candidate queries against each other on the true class logit only. Nothing in
   the cost says "this query should have beaten empty". The `9999*(~inside)` gate
   selects among candidates that already pass, it does not create one.
2. **The inside gate is binary and geometry-based.** A query with an excellent
   polygon but a negative class logit still passes the gate and can win the
   match, while failing the argmax at inference. Measured: best-IoU query per
   suppressed GT has margin **-2.264** (matched **+9.447**). The geometry is
   fine; the class decision is not.

Untested levers that follow, none of which touch the raw `nc+1` argmax / no-NMS
contract:

- put the empty column into the cost: `cost_class = -(p_scores[:, g_cls] - p_scores[:, -1])`
- grade the inside gate by distance outside the polygon (helps small nuclei,
  where a 3 px centroid error flips inside/outside; `no_good_polygon` median
  area is 45.5 px)
- allow GTs with no acceptable owner to stay unmatched instead of being
  force-matched at cost 9999

## 8. What would actually give a clean map (deferred)

Deferred to after the ablation series; recorded so the options are not lost.

1. **Retrain the oracle against a true centroid target** —
   `raycasted/model/train_nulite_seg.py --center-potential-target` trains a
   one-channel max-of-per-instance Gaussian. One maximum per nucleus, amplitude
   encodes size, touching nuclei separate naturally. Cleanest fix; costs one seg
   pre-train. Note the current T8 oracle has `oracle_kind: None` (older
   binary CE+Dice format), so the seed is `logit(fg) - logit(bg)` — a
   foreground-occupancy field with no interior gradient, which is why it is
   plateau-shaped.
2. **Keep the EDT but drop per-instance normalisation** and add centre
   weighting, so peak height encodes size.
3. **Feed the peak offset, not just the peak value** — e.g. add columns for
   "distance to nearest local maximum" and "value of that maximum". This tells
   the head *where* to look, which is the information it is missing, and needs
   no retraining.

## Artifacts

- Probe scripts: `/tmp/opencode/rc_plateau_probe.py` (local-max vs CC),
  `/tmp/opencode/rc_feat_ceiling.py` (257-d capture), plus the probes recorded in
  the two companion documents.
- Results: `/tmp/opencode/rc_full/plateau_t50.json`,
  `/tmp/opencode/rc_full/plateau_t50.log`.
- `/tmp` is a RAM-backed tmpfs; these caches are disposable and the numbers
  above are the record.
