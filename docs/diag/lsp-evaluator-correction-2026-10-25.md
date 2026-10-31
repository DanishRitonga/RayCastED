# Corrected LSP evaluation and query-failure audit — 2026-10-25

The held-out Fold3 evaluator now follows LSP-DETR's test-time mask construction:
PIL polygon filling followed by per-pixel ownership of overlaps using the
greatest Euclidean interior distance. Previously, it filled with OpenCV and
gave overlapping pixels to the largest polygon first. The evaluator also no
longer auto-enables polygon NMS for duplicate-trained checkpoints. Its default
uses the model's raw `nc+1` argmax decision, without a confidence threshold or
deduplication. Overlap ownership constructs masks **for scoring only**; it does
not alter the model's query outputs or inference contract. The old rule remains
available only as `--compare-legacy-overlap` for labeled comparison.

The three `last.pt` checkpoints below were evaluated on the same genuinely
held-out Fold3: 2,722 tiles and 66,654 original dense PanNuke GT instances.
These are single-fold measurements. `output/f1-train/train36` is the older
reference and stopped at epoch 246, whereas V2 Train30 and Train31 stopped at
epoch 332; all used a 600-epoch learning-rate schedule. Thus Train36 is a
**provisional, unmatched control**, not proof that either auxiliary loss caused
the small differences.
This Train36 is distinct from the historical, contaminated
`output/nulite/train36` run.

| Last checkpoint | Whole bPQ | bDQ | bSQ | Whole mPQ | Macro bPQ | Macro mPQ | Centroid P / R | Raw FG queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `output/f1-train/train36` | 0.6274 | 0.7690 | 0.8007 | 0.4445 | 0.6458 | 0.4498 | 0.8555 / 0.7463 | 58,149 |
| `output/v2/train30` (appearance) | 0.6274 | 0.7666 | 0.8023 | 0.4404 | 0.6462 | 0.4444 | 0.8546 / 0.7440 | 58,025 |
| `output/v2/train31` (appearance + presence) | 0.6269 | 0.7678 | 0.8006 | 0.4434 | 0.6446 | 0.4458 | 0.8584 / 0.7442 | 57,784 |

Whole bPQ is the mean of per-image binary PQ; macro bPQ/mPQ average the 19
tissue means. P/R use LSP's 12-pixel centroid match and are different from
polygon-IoU precision/recall. Relative to the previous largest-first results,
the corrected whole bPQ changes by +0.0005 (Train36), +0.0001 (Train30), and
+0.0003 (Train31). The correction is real but does not reverse the conclusion:
appearance gains a little segmentation quality without gaining detection
quality; presence does not recover the detection gap.

## Full-fold nucleus-level diagnosis

For each GT, the audit first does class-agnostic one-to-one polygon-IoU matching
at `IoU > 0.5` using the corrected output masks. For unmatched GT, it inspects
**all raw decoder query polygons before the foreground/empty decision**. A
`good polygon → background` miss has at least one raw query with polygon IoU
above 0.5 but every such query's empty logit beats all foreground logits.
`No good polygon` means no raw query crosses IoU 0.5. `Good FG, unmatched`
covers the small residual from output competition or overlap effects.

| GT / prediction count | Train36 | Train30 | Train31 |
|---|---:|---:|---:|
| GT matched at polygon IoU >0.5 | 49,143 | 48,983 | 48,987 |
| Matched with correct nucleus type | 41,086 | 40,898 | 41,125 |
| Miss: good polygon classified background | 10,626 | 10,764 | 10,623 |
| Miss: no good raw query polygon | 6,764 | 6,781 | 6,949 |
| Miss: good foreground query, unmatched | 121 | 126 | 95 |
| Unmatched evaluated predictions | 9,005 | 9,042 | 8,797 |
| Raw IoU>0.5 duplicates around matched GT (narrow proxy) | 23 | 20 | 9 |

Train30 gains 2,823 GT matches and loses 2,983 versus Train36 (net −160);
Train31 gains 2,847 and loses 3,003 (net −156). On a per-image match count,
Train30 improves/worsens/ties on 831/882/1,009 images; Train31 on
814/881/1,027. The small aggregate difference therefore hides substantial
reshuffling of **which** nuclei are detected. Train30 increases the
good-polygon/background miss cohort by 138. Train31 leaves it essentially
unchanged (−3) and increases no-good-polygon misses by 185. Neither treatment
reduces the major failure cohort.

The misses are concentrated in small nuclei. With GT area ≤200 pixels
(16,747 nuclei), Train36 matches 6,352 (37.9%), suppresses a good polygon for
5,214 (31.1%), and has no good polygon for 5,150 (30.8%). Its match rate is
83.5% for areas 201–611 and 90.3% above 611. Median GT area is 432 pixels
for matched nuclei, 206 for good-polygon/background misses, and 64 for
no-good-polygon misses. Among the 1,057 Dead-class nuclei, only 551 (52.1%)
have a polygon match and 331 (31.3%) are also typed correctly. Train31 raises
small-nucleus matches by just 14, while losing 151 medium and 19 large matches;
it loses 72 matches among nuclei with another GT centroid within 24 pixels.

The background suppression is usually decisive rather than a near tie:
Train36's best good-query foreground-minus-empty logit margin has median
−2.325 (10th/90th percentiles −6.519/−0.389); only 12.6% of suppressed GTs
have a best margin above −0.5. Their best raw polygon has median IoU 0.708.
The corresponding median margins are −2.549 for Train30 and −2.537 for
Train31. These observations support a foreground/empty decision problem for
many viable small-nucleus queries, alongside a separate geometry/coverage
problem for the smallest nuclei. They do not by themselves identify whether
the cause is representation, matching supervision, or calibration.

The duplicate figure counts only raw foreground predictions whose polygon
IoU exceeds 0.5 with an already matched GT. It is not a complete false-positive
taxonomy. The 9,000-ish unmatched predictions and wrong-type matches should
not be conflated with that narrow duplicate cohort.

## Reproduction and artifacts

Run `raycasted/evals/eval_lsp.py` on each checkpoint with `--data-dir` set to
its `transformed/test` tree, the Fold3 parquet, `--device 0`, and
`--save-predictions`. Run `raycasted/evals/audit_lsp_cohorts.py` on each cache
with `--max-images 2722`. The commands used `--batch 8`, `--workers 0`, and
no confidence threshold or NMS. Logs, raw-query NPZ caches, and per-GT
CSV/JSON audits are under `output/diagnostics/lsp-corrected-2026-10-25/`.

Validation: `tests/phase_6/` passed (30 tests); the overlap resolver has
focused parity tests against a full PIL + distance-transform reference;
Ruff on modified Python files and `git diff --check` passed. The repository-wide
Ruff check still reports 241 unrelated existing errors. The next controlled
test still needs a fresh zero-weight Train36-style control stopped at epoch 332
from matched initialization and seed before attributing differences to appearance or
presence supervision.
