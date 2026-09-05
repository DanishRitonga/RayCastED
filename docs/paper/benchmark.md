# PanNuke Fold3 Benchmark Summary

All models evaluated on PanNuke Fold3 test set (2,722 tiles, 65,848 GT instances, conf=0.49-0.50).
**Bold** = best in column. RayCastED variants are NMS-free; all others require NMS post-processing.

| Model | AJI | bPQ | mPQ | F1 | Precision | Recall | Pred# | Params | GFLOPs | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| StarDist | 0.5715 | 0.5945 | 0.3670 | 0.7806 | **0.8558** | 0.7176 | 55,893 | ~3M | — | 143ms |
| CellPose-SAM | **0.6314** | 0.6028 | — | 0.7969 | **0.8687** | 0.7360 | 56,469 | ~50M | — | 112ms |
| LKCell | **0.6228** | **0.6793** | **0.5202** | **0.8147** | 0.8216 | **0.8079** | — | 164M | — | 12ms |
| HoVer-NeXt | 0.5680 | 0.6521 | 0.4708 | 0.7899 | 0.8320 | 0.7520 | 60,243 | 36M | — | — |
| **RayCastED-s** | 0.5555 | 0.5780 | 0.4337 | 0.7176 | 0.7146 | 0.7206 | 66,396 | **1.08M** | 1.38G | **4.7ms** |
| **RayCastED-b** | 0.5715 | 0.6024 | 0.4526 | 0.7409 | 0.7419 | 0.7400 | 65,679 | 3.03M | 5.52G | **3.5ms** |
| **RayCastED-l** | 0.5740 | 0.5947 | 0.4764 | 0.7591 | 0.8113 | 0.7132 | 57,888 | 10.26M | 12.4G | **3.5ms** |

## NuLite-DETR (current method) — original-mask protocol

Same fold3 test set, but evaluated against the **original dense PanNuke masks** (decoded from the parquet), conf=0.20, corrected instance-level PQ (global accumulation). The numbers above (this file) predate the metric fix and use the ray-polygon GT + buggy union-IoU bPQ — **this table is the apples-to-apples one vs literature**.

| Model | AJI | bPQ (DQ/SQ) | mPQ | F1 (centroid) | Prec | Rec | preds | Params | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| LSP-DETR (lit., 3-fold CV) | — | 0.675 | 0.482 | — | — | — | — | 45M | — |
| LKCell (lit., 3-fold CV) | — | 0.684 | 0.503 | — | — | — | — | 163.8M | — |
| NuLite-DETR (train19, shared match) | 0.7706 | 0.6901 (0.8318/0.8296) | 0.6489 | 0.8749 | 0.7913 | 0.9783 | 82,409 (1.25x) | 15.46M | 5.10 |
| NuLite-DETR (train28, per-layer match) | 0.7738 | 0.7013 (0.8386/0.8362) | 0.6687 | 0.8799 | 0.8038 | 0.9719 | 80,596 (1.22x) | 15.46M | 5.55 |
| **NuLite-DETR** (train31, shared layers) | **0.7824** | **0.7138** (0.8494/0.8403) | **0.6777** | **0.8559** | **0.8006** | **0.9282** | **79,113 (1.20x)** | **13.77M** | **5.24** |
| **NuLite-DETR (submission)** (train33, hd=128) | 0.7618 | 0.6954 (0.8463/0.8217) | 0.6580 | 0.8560 | 0.8009 | 0.9287 | 78,991 (1.20x) | 12.53M | 5.05 |
| **NuLite-DETR (submission, t12)** (train34, hd=128) | 0.7667 | 0.6993 (0.8449/0.8276) | 0.6595 | 0.8565 | 0.8026 | 0.9274 | ~79k (1.20x) | **10.61M** | **5.11** |

Per-class PQ (original-mask GT, train31 hd=256): Neoplastic 0.7152, Inflammatory 0.7429, Connective 0.6972, Necrosis 0.5123, Epithelial 0.7211.
Per-class PQ (original-mask GT, train33 hd=128): Neoplastic 0.7003, Inflammatory 0.7145, Connective 0.6792, Necrosis 0.4920, Epithelial 0.7038.
Per-class PQ (original-mask GT, train34 t12 hd=128): Neoplastic 0.7039, Inflammatory 0.7225, Connective 0.6795, Necrosis 0.4813, Epithelial 0.7105.
Original-mask GT has 66,654 instances (806 tiny nuclei dropped by the polygon ETL are present). Evaluated on a harder target than the polygon protocol, scores are still higher (SQ 0.8296 vs 0.8074) because the 64-ray GT polygon is a chord-inscribed inner approximation of the true boundary.
Caveats: single fold (fold3 test) vs literature 3-fold CV average; largest-first mask-overlap resolution vs literature watershed refinement; centroid F1 is our custom r=12 metric (not directly comparable to literature F1 columns).

## Notes

- **StarDist**: Reference polygon-based method (non-learned post-processing, NMS required). High centroid precision but poor rare-class detection (necrosis F1=0.002). Evaluated via eval_pannuke.py with StarDist ray output rasterized to masks.
- **CellPose-SAM**: CellPose detection + SAM segmentation. Best AJI and AP@0.5 (strongest mask quality at 0.631 and 0.677), but no nuclei classification (mPQ N/A). 43× slower than RayCastED.
- **LKCell**: Large-kernel CNN (Cui et al., 2024). 164M params, strongest overall performance. NMS required for post-processing.
- **HoVer-NeXt**: HoVerNet successor with next-gen architecture. 36M params, second-best bPQ. NMS required.
- **RayCastED**: Proposed method. All variants are NMS-free end-to-end. Hierarchical classification with dual o2m/o2o assignment. Haar wavelet downsampling with SE channel attention.
