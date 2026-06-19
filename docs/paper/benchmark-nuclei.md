# PanNuke Fold3 Nuclei Class Breakdown — F1 Score

All models evaluated on PanNuke Fold3 test set. **Bold** = best in row.
CellPose-SAM excluded (no nuclei classification). F1 is centroid-based, class-matched (centroid ≤12px AND class-correct).

| Class | StarDist | LKCell | HoVer-NeXt | RayCastED-s | RayCastED-b | RayCastED-l |
|---|---|---|---|---|---|---|
| Neoplastic | 0.610 | **0.723** | 0.695 | 0.659 | 0.677 | 0.692 |
| Inflammatory | 0.564 | **0.684** | 0.665 | 0.578 | 0.585 | 0.624 |
| Connective | 0.471 | **0.603** | 0.596 | 0.509 | 0.532 | 0.552 |
| Necrosis | 0.002 | 0.383 | **0.409** | 0.394 | 0.392 | 0.395 |
| Epithelial | 0.354 | 0.723 | **0.725** | 0.630 | 0.657 | 0.692 |

## Notes

- **StarDist** has extreme class imbalance: necrosis F1=0.002 (essentially zero detection). Its high global F1 (0.781) masks severe rare-class failure.
- **RayCastED** is the only model family where necrosis F1 consistently exceeds 0.390 — the hierarchical classification design properly handles rare classes.
- **LKCell** leads on Neoplastic/Inflammatory/Connective due to 164M params, but necrosis (0.383) is slightly below RayCastED-s (0.392).
- **HoVer-NeXt** wins Epithelial (0.725) and edges RayCastED on most classes despite 12× more parameters.
