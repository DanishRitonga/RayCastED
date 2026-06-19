# Hierarchical Dual-Head Classification

## Architecture

The o2o classification branch is split into two independent heads operating in cascade:

```
                        ┌─────────────────┐
Backbone → Neck → P2/P3/P4 Features ──────┤── stop-grad ──┐
                        │                  │               │
                        │          ┌───────▼────────┐  ┌───▼──────────────┐
                        │          │  Binary Head   │  │  Class Head      │
                        │          │  Conv2d → 1ch  │  │  Conv2d → nc ch  │
                        │          │  FG/BG logits  │  │  Class logits    │
                        │          └───────┬────────┘  └───┬──────────────┘
                        │                  │                │
                        │          Focal BCE on ALL     BCE (1-vs-rest) on
                        │            5376 anchors      fg_mask anchors only
                        │                                (TAL-assigned fg)
                        │
                        │  Inference: final_conf = σ(binary) × σ(class)
                        │  Hard gate:  σ(binary) < 0.01 → suppress class scores
                        └───────────────────────────────────┘
```

**Binary Head**: A 4-layer conv stack (DWConv → Conv → DWConv → Conv) ending in `Conv2d(c3, 1)` producing a single fg/bg logit per anchor. Trained with **focal BCE** on all 5,376 anchors — fg anchors get target=1, bg anchors get target=0. This gives the binary head a strong fg/bg discrimination signal from ~98.7% of anchors.

**Class Head**: Same 4-layer conv stack ending in `Conv2d(c3, nc)` producing nc class logits per anchor. During **training**, the class head forward-propagates all 5,376 anchors, but only anchors in the TAL-assigned `fg_mask` receive class BCE loss. During **inference**, class logits for all anchors are multiplied by `sigmoid(binary)` — anchors with `sigmoid(binary) < 0.01` are hard-suppressed to zero. This decouples class decisions so rare classes (e.g., necrosis at ~1% of nuclei) get independent 1-vs-rest binary decisions rather than being diluted by a softmax denominator.

**Stop-Gradient**: The binary head's input features are detached (`x.detach()`). The class head's BCE gradients flow through the *undetached* backbone features. This prevents the binary head's focal BCE on 5,376 anchors from flooding the backbone with gradient noise, while preserving the class head's ability to shape features via inter-class discrimination.

## Inference

```
final_conf = sigmoid(binary_logit) × sigmoid(class_logit)
```

Scores are clamped by a hard gate: if `sigmoid(binary) < 0.01`, the anchor's class scores are suppressed to zero. This prevents the class head from emitting confident class predictions for anchors the binary head has already identified as background.

Following inter-scale competition, `final_conf` is multiplied by the softmax weight from cross-scale competition (P2/P3/P4).

## Motivation

Standard single-head classification in FCN detectors (YOLO, FCOS) suffers from a **gradient starvation** problem in the o2o branch: with `topk2=1`, only ~28 anchors per image are foreground (0.5% of 5,376). The remaining 98.7% produce either zero cls gradient (for non-assigned anchors) or weak focal loss bg gradients. This starves the cls head of sufficient fg/bg discrimination signal.

The hierarchical design solves this by:

1. **Binary head gets all 5,376 anchors**: Every anchor is either fg (assigned) or bg (everything else). This gives the binary head ~5,376 classification targets per image, compared to ~28 fg + ~84 sampled bg in the standard design — **42× more gradient signal for fg/bg learning**.

2. **Class head only sees fg anchors**: The nc-way class decision is performed on features that have already been filtered to foreground candidates. This simplifies the task from "classify among nc+1 options (including bg)" to "classify among nc options."

3. **Decoupled gradients**: Stop-gradient prevents the binary head's massive fg/bg gradient signal from dominating backbone updates, while allowing class-discriminative features to flow through the undetached path.

## Comparison to Related Work

### FCOS Centerness (Tian et al., ICCV 2019)
FCOS uses a separate centerness branch that predicts `sqrt(min(l,r)/max(l,r) × min(t,b)/max(t,b))` — a geometric heuristic for "how centered is this point." The centerness score multiplies with class confidence at inference.

**Difference**: Centerness is a fixed geometric prior, not learned. It encodes a specific assumption (objects have one center) and provides no gradient for fg/bg discrimination. Our binary head learns what constitutes foreground through focal BCE optimization, producing a data-driven gate that adapts to the training distribution.

### RPN in Two-Stage Detectors (Faster R-CNN, Ren et al., NeurIPS 2015)
Two-stage detectors use a Region Proposal Network (RPN) to generate foreground proposals, then a separate classifier head operates on the proposed regions.

**Difference**: The RPN operates on shared backbone features in a separate network pass. Our binary head is a deepcopy of the cls head architecture applied to the same features with a stop-gradient — it's single-pass with minimal parameter overhead (1+nc output channels vs standard nc output channels).

### YOLOX Decoupled Head (Ge et al., 2021)
YOLOX separates cls and reg into independent branches with separate convolutions.

**Difference**: Decoupled head separates task types (cls vs reg). Our hierarchical head separates the classification *problem* into two sub-problems (fg/bg vs inter-class). Both branches still do classification, but at different levels of the decision hierarchy.

## Implementation

| Parameter | Value |
|-----------|-------|
| `hierarchical_cls` | true |
| `hierarchical_cls_detach` | true |
| `hierarchical_binary_threshold` | 0.01 |
| Binary loss | Focal BCE, γ=2.0, α=0.75, ALL anchors |
| Class loss | BCE (1-vs-rest), fg anchors only |
| Binary head | `Conv2d(c3, 1)` — replaces `Conv2d(c3, nc)` in cv3 |
| Class head | `Conv2d(c3, nc)` — standard cv3 |
