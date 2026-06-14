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

## Notes

- **StarDist**: Reference polygon-based method (non-learned post-processing, NMS required). High centroid precision but poor rare-class detection (necrosis F1=0.002). Evaluated via eval_pannuke.py with StarDist ray output rasterized to masks.
- **CellPose-SAM**: CellPose detection + SAM segmentation. Best AJI and AP@0.5 (strongest mask quality at 0.631 and 0.677), but no nuclei classification (mPQ N/A). 43× slower than RayCastED.
- **LKCell**: Large-kernel CNN (Cui et al., 2024). 164M params, strongest overall performance. NMS required for post-processing.
- **HoVer-NeXt**: HoVerNet successor with next-gen architecture. 36M params, second-best bPQ. NMS required.
- **RayCastED**: Proposed method. All variants are NMS-free end-to-end. Hierarchical classification with dual o2m/o2o assignment. Haar wavelet downsampling with SE channel attention.
