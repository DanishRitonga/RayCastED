# HANDOVER.md — for the next agent (2026-09-07)

Branch: `feature/nulite-raycast-seg` — HEAD `3d582e6`. Working tree clean.

---

## OBJECTIVE

Match LSP-DETR's evaluation protocol and close the honest quality gap on PanNuke.
Reference numbers (LSP-DETR paper Table 3, per-tissue macro, 3-fold CV, final ckpt):
**LSP-DETR bPQ 0.675 / mPQ 0.482**, LKCell 0.684/0.503, CellViT 0.679/0.498.

Our current honest standing (LSP protocol, single held-out fold, best.pt):
- Locked t12-sep recipe (rotation-1, fold1→fold3): macro **bPQ 0.6202 / mPQ 0.4293**
- train19 s12 old-decoder (fold3→fold2): macro **bPQ 0.6148 / mPQ 0.4223**
- Gap ≈ 5-6 bPQ pts and 5-6 mPQ pts. **Recall-limited** (P ~0.89, R ~0.69).

User directive (verbatim): *"we will use the numbers in LSP-DETR paper for reference.
thus, we'll only need to match their protocol"* and *"our best performing model is
actually outclassed by lsp-detr and lkcell? we'll need to somehow improve it"*.

User instinct under investigation: *"does ∅ decision/threshold even significantly help?
maybe we need architecturally different approach"* — see NEXT STEPS.

---

## CRITICAL CONTEXT: CONTAMINATION (do not regress on this)

**ALL historical test-set numbers (train18–train36) in `docs/runs/nulite-detr/*.md`,
`docs/paper/benchmark.md`, and AGENTS.md experiment log are INVALID** — they were
measured on the *training* fold:

- All NuLite DETR runs trained on `output/nulite/transformed/train` = **fold3** tiles
  (args.yaml `data=.../nulite/transformed/data.yaml`).
- All historical test evals used `--data-dir output/pannuke_64/transformed/test` =
  **fold3** = identical 2722 tiles as the training tree → 2722/2722 overlap.
- Validation was unaffected (ran on fold1), so best.pt selection is legit; only the
  final "test" reporting step was wrong.

**Folds** (identical to LSP-DETR's HF `RationAI/PanNuke` splits): fold1=2656, fold2=2523,
fold3=2722 images. Categories 0-4 (Neoplastic/Inflammatory/Connective/Dead/Epithelial),
19 tissues. Parquets: `raycasted/data/dataset/PanNuke/data/fold{1,2,3}-00000-of-00001.parquet`
(categories already 0-4, ROI id == row order).

**Protocol that must be used for any number compared to literature:**
- GT = dense original masks from the fold parquet (NOT 64-ray polygons).
- No confidence threshold: keep a query iff argmax over the (nc+1) logits is a real
  class (drop the no-object ∅ column) — this is LSP-DETR's exact decision.
- **bPQ = per-image binary PQ then mean over images** (IoU>0.5, Hungarian match,
  dq=tp/(0.5·P+0.5·M), sq=matched-IoU/tp, pq=dq·sq). Empty-GT images skipped.
- **mPQ = per-image mean over GT-present classes**, then mean over images.
- Centroid F1 radius 12, global tp/fp/fn.
- Per-tissue accumulation; reference row = macro mean over 19 tissues. LSP uses the
  **final (last) checkpoint**, not best-val. Best==last for rotation-1 (plateau).
- Tool that does exactly this: **`raycasted/evals/eval_lsp.py`** (committed 3d582e6).
  Run: `uv run python -m raycasted.evals.eval_lsp --weights <best.pt> --data-dir <test tiles dir> --parquet <fold parquet> --batch 16 --device 0`.

**`eval_pannuke.py` uses a DIFFERENT (older) protocol** — global-PQ accumulation +
conf>0.20 filter + polygon GT — and `docs/*` results written before the metric fix
(commit e2f73ea) use yet another (union-IoU) protocol. **Do not trust absolute numbers
in docs/; trust only `eval_lsp.py` outputs on genuinely held-out folds.**

---

## ARCHITECTURE (locked in `docs/paper/architecture.md`, commit 16f641d)

NuLite-DETR = convert NuLite-style instance segmentation into an NMS/post-proc-free
end-to-end star-convex-polygon detector:
Image(256) → FastViT encoder (`timm`, `features_only`) → NuLite-parity upsample
decoder → {b3(st4), b4(st8), b5(st16)} multi-scale feats + b1(st1); decoder0(_ConvBlock)
on input; `xt=cat([decoder0(x), b1])` → NPHead → seed map (B,1,256,256 raw logits).
DETR head (`NuLiteDETRDecoder`, subclass of ultralytics RTDETRDecoder): LSP-DETR fixed
20×20=400 grid queries; query content = grid_sample of projected P2 at grid points p0;
3 deformable decoder layers (optionally weight-shared); per-layer polygon MLP→66-dim
(cx,cy,64 rays; iterative refinement + SAP-DETR tanh drift confinement) + class head
5+∅ (focal no-object). conf=1-P(∅), cls=argmax over 5.

Seed map dual-use (all proven beneficial in the seed-ablation arc): **S** seg loss
(lambda_seg·L1 vs InstanSeg per-instance-EDT target, target=(edt−0.5)·15), **F** extra
feature channel concatenated to b3/b4/b5 before input_proj, **Q** raw seed logit
concatenated into enc/dec score-head inputs (`_score`).

Loss/matching: per-image Hungarian (focal class + |Δrays|+|Δxy| + (1−pIoU) +
cost_inside=10 for centroid-outside-GT) + per-layer "look-forward-twice" matching +
FCN-parity regression terms (xy Huber·500, log-space ray L1·14, pIoU·3, smooth·0).

Key model files (all under `raycasted/model/`): `blocks/nulite.py`
(NuLiteEncoderDecoder/_ConvBlock/NPHead/PANBottomUp/SAMConditioner),
`blocks/nulite_detr.py` (NuLiteDETRDecoder + encode/decode_polygon + `_rank_causal_mask`,
+ `_grid_decoder_input`/`_learned_decoder_input`/`_seed_decoder_input` query modes),
`nulite_detr_loss.py` (RayCastHungarianMatcher + NuLiteDETRLoss),
`nulite_detr_model.py` (NuLiteRayCastDETRModel: encoder_decoder+decoder0+np_head+detr;
eval→(B,400,68) pixel-scaled [cx,cy,rays,conf,cls]), `train.py` (_get_nulite_detr_model,
get_validator loss_names, validate() warmup guard).

Full locked-recipe config knobs in `main/pannuke.yaml` training section:
`architecture=nulite_detr, nulite_variant=fastvit_t12, pretrained=true, lambda_seg=1.0,
seed_map_target=true, detr_hd=128, detr_d_ffn=1024, detr_ndl=3, detr_shared_layers=true,
detr_separable_decoder=true, detr_per_layer_match=true, detr_grid_size=0.05 (→400 q),
detr_query_selection=grid, detr_seed_feature=true, detr_seed_in_content=true,
detr_no_object=true, detr_no_object_weight=2.0, detr_cost_inside=10.0, detr_mds=false,
fitness_warmup_epochs=75, amp=false, patience=300, warm_restarts=false,
inference_conf=0.20`. → ~7.85M params.

---

## 3-FOLD CV PROTOCOL (user-confirmed workflow)

Each CV model trains on ONE fold. Committed scheme: `main/pannuke.yaml`
`datasets.PanNuke.split_args.split_map`. Current committed value (rotation-1, commit
688427d): fold1=train, fold2=val, fold3=test.

For each rotation: edit split_map + run pipeline with a **fresh --output dir**
(user: *"we will simply set the output flag to a different output"*).

- Train: `uv run python -m raycasted.pipeline --config main/pannuke.yaml --output <fresh dir> --stage train --epochs 600 --batch 16 --device 0`
  (runs land `<dir>/trainNN`). Do NOT use --stage all unless ETL tiles are absent.
- Eval (held-out fold test tiles): `uv run python -m raycasted.evals.eval_lsp --weights <dir>/.../weights/best.pt --data-dir <dir>/transformed/test --parquet raycasted/data/dataset/PanNuke/data/<testfold>-00000-of-00001.parquet --batch 16 --device 0` (~5 min GPU, 700s timeout, log to /tmp/opencode/).
- User runs training on cuda:0 RTX 5070 Ti; I (agents) prep config + verify + may run evals. **Do NOT launch training.**

Rotation-1 DONE: `output/f1-train/train/weights/{best,last}.pt` (locked t12-sep 7.85M,
fold1→fold3). Test tree `output/f1-train/transformed/test` (fold3, 2722 tiles).
LSP-protocol numbers (above); full log `/tmp/opencode/eval_lsp_f1train.log`;
run doc committed at `docs/runs/nulite-detr/cv-lsp-protocol.md`.
Rotations 2/3 NOT yet run (fold2-train and fold3-train of the locked recipe).

---

## WHAT WE TRIED + RESULTS (summarized; full details in docs/runs + compressed history)

**FCN→DETR pivot.** Original YOLO26-FCN RayCastED suffered same-scale duplicate
overprediction. Built NuLite-DETR (above) — query grid + 1:1 Hungarian eliminates
duplicates (overprediction 6.1x → 1.2-1.3x) but changed the whole game to
winner-selection/discrimination.

**Seed-branch ablation arc (fold3-contaminated test, RELATIVE ranking still valid):**
S+F+Q > S > S+F > none > Q. Seg loss is the main seed contributor; Q alone harmful but
unlocks F in the trio. → locked S+F+Q.

**Head-slimming ladder** (selected on contaminated folds — clean evals later showed it
HURT): shared layers (train31) won on contaminated test; separable decoder + t12 + hd128
enabled sub-10M (down to 7.85M) but slimming reduces discrimination. Clean leaderboard
(fold2, OLD eval_pannuke protocol): train19 s12 old-decoder bPQ 0.6123 best; all
post-train19 "improvements" (per-layer-match train28 0.5059, decoder parity, shared
layers, hd128, t12, separable → 0.51) actually REGRESSED. train19 recipe = s12 OLD
(pre-parity) decoder, MDS off, S+F+Q seed, cost_inside, hd256, NO per-layer-match,
15.46M.

Under the LSP protocol the locked t12-sep and train19 recipes are within noise of each
other (~0.62 macro bPQ).

**Recall ceiling diagnostic (partial)**: on 400 fold3 tiles / 9753 GT, rotation-1 model:
72.3% of GT have a *fired* (∅-passed) query within 12px; 99.9% have *any* of the 400
queries within 12px (caveat: 20×20 grid geometry makes ~everything covered). → The
recall gap is concentrated in the ~27.7% of GT whose nearest query is **∅-suppressed**,
i.e. sits at the class/∅ decision boundary. Diagnostic script (unfinished, needs a
conclusion run): `/tmp/opencode/recall_ceiling_diag.py`. The key unknown: for
suppressed-only GT, is the nearest query a correctly-localized polygon that the gate
wrongly drops (→ lower `detr_no_object_weight`/retrain helps) or a poorly-placed query
(→ gate tuning dead, need architecture)?

**What did NOT work historically** (AGENTS.md gotchas): soft targets/QFL, DINO
denoising, quality head, PSS head, self-attention variants, AIFI, higher topk2. All on
the old FCN; the DETR line superseded those.

---

## NEXT STEPS (in suggested order)

1. **Finish the recall-ceiling diagnostic** (build mode, ~2 min GPU): in
   `/tmp/opencode/recall_ceiling_diag.py`, compute the IoU of the *nearest query polygon*
   for the ∅-suppressed-only GT. If most are IoU>0.5 TPs → the ∅ gate is wrongly killing
   good detections → fix by retraining rotation-1 with a **lower `detr_no_object_weight`
   (2.0 → ~0.5)** and re-eval under eval_lsp. If most suppressed queries are off-center /
   low-IoU → gate tuning is a dead end.

2. **Architecture path (likely needed)**: the discrimination capacity is the gap axis
   vs LSP-DETR (Swin-V2-T fine-tuned, d=384, 6 layers, 45M vs our t12 sep d128 3 shared
   layers 7.85M, ImageNet-only encoder with no fine-tuning phase). Restore capacity on a
   CLEAN rotation: s12 or fastvit_sa12/sa24, unshared layers, hd 256-384, and/or a
   proper backbone-fine-tune schedule (freeze 30 ep @10% lr like LSP). One clean 3-fold
   rotation of the bigger model answers whether 0.675 is reachable.

3. **Finish rotations 2/3** of whichever recipe is chosen → report mean±std across the 3
   test folds (final/last ckpt) → this is the number for the paper vs LSP Table 3.

4. **Update the record**: docs/runs/nulite-detr/{train18..train36}.md + benchmark.md +
   AGENTS.md contain contaminated absolute numbers — add a warning header or rewrite
   with clean fold2/3 numbers before any paper use. (AGENTS.md still describes the old
   FCN architecture / raycasted-v1 and has NO mention of NuLite-DETR / eval_lsp / the
   contamination / the locked architecture.)

---

## DURABLE INFRA FACTS

- Repo: `feature/nulite-raycast-seg` @ `3d582e6`. Machine has GPU (cuda:0 RTX 5070 Ti);
  USER runs training interactively. ruff clean expected on edited files
  (`uv run ruff check`); tests `tests/phase_10/test_nulite_integration.py` +
  `test_nulite_detr_integration.py` 12/12. GFLOPs printed by eval_pannuke is a HARDCODED
  placeholder — use ms/img.
- Data: tiles `output/pannuke_64/transformed/{train,val,test}` (fold1/2/3 = 2656/2523/2722,
  keys image/annotations/content_h/content_w/tissue) and `output/nulite/transformed/*`
  (the CONTAMINATED historical tree: train=fold3, val=fold1, test=fold2).
- Reference paper PDFs/text in /tmp/opencode/: lsp_detr.pdf/txt, lkcell.pdf/txt,
  cellvit.pdf/txt, hover_net.pdf/txt, pannuke_ext.pdf/txt, pannuke_readme.md.
  LSP-DETR source clone: `/tmp/opencode/lsp-detr` (metrics/binary_panoptic_quality.py,
  panoptic_quality.py, data/pannuke.py, configs/experiment/PanNuke.yaml are authoritative
  for the protocol). `pannuke_readme.md` = HF RationAI/PanNuke card.
- Era worktrees for loading old checkpoints under era code: `/tmp/opencode/wt-train18`
  (b7444b6, old-decoder era) and `/tmp/opencode/wt-train28` (7d3368e). eval_lsp.py was
  copied into wt-train18 already.
- Old checkpoints: `output/nulite/train{18..36}/weights/{best,last}.pt` (trained fold3;
  train19 best clean). NuLite source clone at /tmp/nulite_src if needed.
