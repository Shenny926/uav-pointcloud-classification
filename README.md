# UAV Point Cloud Classification — Photogrammetric Data Acquisition and Analysis (TUM)

Cross-epoch point cloud classification: training classifiers on 2024/2025 UAV point clouds
(Roggensteiner Str., Emmering) and applying them to our 2026 field acquisition.

**Presentation:** 24 July 2026 · **Report due:** 31 August 2026

---

## Objective (from assignment)

- Use point clouds from a different epoch (2024/2025) to train classifiers and classify our own 2026 acquisition.
- Investigate whether results differ between classifiers or between training datasets.

## Classes

| Label | Class |
|---|---|
| 1 | building |
| 2 | road |
| 3 | vegetation |

(Cars were considered but dropped — no labeled car data available.)

---

## Data (NOT in this repo — too large for GitHub)

Located on Shenjian's machine, ask if you need transfers:

```
E:\shenj\Documents\pointcloud_2024_2025\
    2024\Nadir_withRTK-dense_point_cloud_2024[_buildings|_roads|_vegetation].laz
    2025\2025_Nadir-dense_point_cloud_[buildings|roads|vegetation].laz
    training_data.npz              (merged, 51.1M points)
    training_data_subsampled.npz   (300k per class, 900k total, seed=42)
    training_features.npz          (v1 features)
    training_features_v3.npz       (v3 features)
    rf_classifier.joblib           (v1 model)
    rf_classifier_normalized.joblib(v2 model)
    rf_classifier_v3.joblib        (v3 model)
    greenness_norm_stats.json

E:\shenj\Documents\test_set_2026\
    PhotoLab_Group1-dense_point_cloud_Test_set.2026.laz   (11.8M points, raw)
    PhotoLab_Group1_2026_classified.laz      (v1 output)
    PhotoLab_Group1_2026_classified_v2.laz   (v2 output)
    PhotoLab_Group1_2026_classified_v3.laz   (v3 output)
```

The per-class 2024/2025 files were manually segmented (teammate) in CloudCompare;
the full clouds have classification field = 0 (unclassified) straight from Pix4Dmatic.

---

## Pipeline (scripts in execution order)

| Script | Purpose |
|---|---|
| `inspect_laz.py` | Open a LAZ, print point count, dimensions, classification values |
| `build_training_data.py` | Load 6 per-class files (3 classes × 2 years), tag labels, merge → `training_data.npz` |
| `subsample_training_data.py` | Random 300k points per class (seed=42) → `training_data_subsampled.npz` |
| `extract_features.py` | v1 features (see below) → `training_features.npz` |
| `train_classifier.py` | Train/evaluate Random Forest (v1) → `rf_classifier.joblib` |
| `classify_2026.py` | Apply v1 model to 2026 cloud, write predictions into LAS classification field |
| `diagnose_features.py` | Compare greenness/z distributions between training and 2026 data |
| `retrain_normalized.py` | v2: retrain with globally z-score-normalized greenness |
| `classify_2026_v2.py` | Apply v2 model to 2026 (greenness normalized per-dataset) |
| `extract_features_v3.py` | v3 features: local (neighborhood-relative) greenness |
| `train_classifier_v3.py` | Train/evaluate v3 model |
| `classify_2026_v3.py` | Apply v3 model to 2026 |

**Dependencies:** `pip install laspy[lazrs] scikit-learn` (Python 3.13, numpy included via deps)

### Features (per point, K=20 nearest neighbors)

Geometric features from PCA eigenvalues (λ1 ≥ λ2 ≥ λ3) of the local neighborhood covariance:

- linearity (λ1−λ2)/λ1, planarity (λ2−λ3)/λ1, sphericity λ3/λ1
- verticality: 1 − |n_z| of the smallest-eigenvector normal
- roughness: √λ3
- height_range: max−min z within neighborhood

Plus: absolute z, intensity, and a color feature (varies by version, see below).

---

## Results

### v1 — Raw greenness  `g/(r+g+b)`

- Held-out test (20% of training data): **94% accuracy** (building F1 0.96, road 0.93, vegetation 0.93)
- Feature importance: greenness 34.6%, z 19.5%, verticality 14.8%, height_range 13.4%; intensity 0.0 (dead feature — worth noting in report)
- **2026 prediction: building 34.4% / road 54.9% / vegetation 10.7%**
- Visual check in CloudCompare: buildings and roads plausible; tree canopies speckled/misclassified — vegetation clearly under-predicted.

### Diagnosis (diagnose_features.py)

- z distributions consistent between training and 2026 → coordinate/elevation reference is NOT the problem.
- Greenness distribution shifted upward in 2026 (overall mean 0.3833 vs 0.3548 in training).
  Training class means: building 0.30, road 0.34, vegetation 0.42.
  2026 predicted-building points average 0.376 — i.e. ordinary 2026 surfaces land where
  training vegetation used to start. Classic cross-epoch radiometric shift
  (lighting/exposure/Pix4Dmatic color processing differences between flight years).

### v2 — Globally normalized greenness (z-score per dataset)

- Same 94% on held-out split (expected — linear rescaling doesn't change within-dataset separability).
- **2026 prediction: building 35.4% / road 58.0% / vegetation 6.6% — WORSE.**
- Why it failed: global normalization assumes similar class composition between scenes.
  If 2026 genuinely has less vegetation, its scene-wide mean is lower due to composition,
  and normalizing against it distorts rather than corrects.

### v3 — Local relative greenness (point minus neighborhood mean)

- Held-out split: 91% accuracy (slightly lower, acceptable trade for expected robustness).
  Feature importance shifted toward geometry (z 23.5%, height_range 19.3%, roughness 17.9%);
  local_greenness only 11.1%.
- **2026 prediction: building 37.6% / road 61.0% / vegetation 1.4% — MUCH WORSE.**
- Why it failed: vegetation is spatially clustered. A canopy point's neighbors are other
  canopy points, so (point − neighborhood mean) → 0 exactly in vegetation interiors.
  The feature only fires at vegetation boundaries.

### Summary table

| Version | Color feature | Test acc. (within-epoch) | 2026 vegetation % |
|---|---|---|---|
| v1 | raw greenness | 94% | 10.7% |
| v2 | globally normalized | 94% | 6.6% |
| v3 | locally relative | 91% | 1.4% |

**Working conclusion (superseded — see Final Version below):** v1/v2/v3 together are the
debugging narrative: v1 as the initial attempt, v2/v3 as documented robustness investigations
demonstrating that cross-epoch radiometric shift is not trivially correctable — each principled
fix failed for an explainable reason. This directly answers the assignment's "differences in
results" question and is report material, not wasted work. The best-performing model is now the
Final Version below; v1/v2/v3 remain the "why it's hard" story.

---

## Final Version (`final_version/`)

Built by Nicolas, added to the repo after the v1–v3 debugging above. Same core idea (Random
Forest on combined 2024/2025 labeled data, applied to 2026) but a different strategy for the
color-shift problem: instead of picking one recalibrated greenness definition, it feeds the
model several complementary signals at once and corrects for the known vegetation
under-prediction bias at the decision stage rather than only at the feature stage.

**Scripts:**

| Script | Purpose |
|---|---|
| `final_version/build_training_data.py` | Same role as the root script — loads and tags the 6 per-class files |
| `final_version/subsample_training_data.py` | Same role as the root script — 300k/class subsample |
| `final_version/featureextractv3.py` | **13 features**, not 9 — see below. K=30 (not 20) |
| `final_version/classifierv3for2026.py` | Trains + applies the model, with probability-threshold tuning and spatial filtering |
| `final_version/evaluate_classifier.py` | Matches a classified cloud against the labeled 2026 reference files by coordinate → real accuracy/confusion matrix (not yet run — see Known gaps) |

⚠️ **Naming collision:** `featureextractv3.py` and `classifierv3for2026.py` exist both at repo
root (our own v3 attempt) and inside `final_version/` (Nicolas'). They are different files with
different logic — the root versions are the "local relative greenness" attempt documented above;
the `final_version/` ones are the final approach. Don't confuse them when reading the repo.

**Feature set (13, vs. our 9):**
- `z_robust`, `intensity_robust` — robust-scaled (median/IQR) instead of raw
- The same 5 PCA geometric descriptors we used (linearity, planarity, sphericity, verticality, roughness) + height_range
- `height_above_local_min` — height above the lowest point in the local neighborhood, replacing raw elevation. This is the fix for the height-above-ground item in the old Next Steps list.
- **Four color signals instead of one:** `green_ratio` (same formula as our v1 greenness), `excess_green` (2G−R−B chromaticity index, standard in remote sensing), `local_mean_green`, `local_greenness` (our v3 idea, kept as a supplementary feature rather than a replacement)

**Decision-level changes (not just features):**
- Uses `predict_proba` with a tuned vegetation threshold (0.40, not the default 0.5 argmax) — deliberately biases toward catching more vegetation
- 3D voxel-based spatial majority filter as post-processing (cleans up speckle noise) — requires `spatial_filter.py`, see Known gaps

**Result: 2026 prediction — building 18.3% / road 31.9% / vegetation 49.8%** — matches the
site's real composition, a large improvement over v1's 10.7% and v2/v3's worse results.

**Why it worked where v1–v3 didn't:** v1 already included both color and elevation, but put
~35% of its decision weight on a single greenness feature and used raw absolute elevation —
a single point of failure. This version gives the model several independent, complementary
signals instead of one, plus explicitly corrects the known bias at the decision threshold. See
the LaTeX report (`classifier_report.tex`) for the full writeup.

**Known gaps:**
- `spatial_filter.py` (the `voxel_majority` function `classifierv3for2026.py` imports) was not
  provided and is not in this repo. Running the script as-is will fail on that import until we
  get it from Nicolas.
- `evaluate_classifier.py` exists but has not actually been run yet — the 49.8% figure above is
  the predicted class proportion, not a verified accuracy against the labeled 2026 ground truth.
  Running this script is the top item in Next Steps below.

---

## Next steps (priority order)

1. **Run `final_version/evaluate_classifier.py` against 2026 ground truth.** Now the top
   priority — the 49.8% vegetation figure is a predicted proportion, not a verified accuracy.
   The script already exists and matches points against the labeled 2026 per-class files by
   coordinate; it just hasn't been executed yet. Would give a real confusion matrix and
   per-class accuracy for the final model, and ideally for v1/v2/v3 too for a fair comparison.
2. ~~Height-above-ground (HAG) feature~~ — **done** in `final_version/` (`height_above_local_min`).
3. ~~Spatial regularization (majority filter)~~ — **done** in `final_version/` (voxel majority
   filter), but blocked on the missing `spatial_filter.py` dependency — get this from Nicolas.
4. **Optional / report comparisons:**
   - Train on 2024-only vs 2025-only vs combined → the "different training datasets" axis
     of the assignment objective.
   - Second classifier type (e.g. SVM or gradient boosting) on identical features → the
     "different classifiers" axis.
   - Consider dropping intensity (importance 0.0) and absolute z (site-specific) from features.
   - If using any 2026 labels for training/calibration: disclose it and keep a held-out
     portion for evaluation, otherwise the cross-epoch objective becomes circular.

---

## Reproducing

1. Adjust the hardcoded paths at the top of each script to your local data locations.
2. Run in the order listed in the Pipeline table.
3. Subsampling uses `seed=42` — results are reproducible.
4. Feature extraction on the full 2026 cloud (11.8M points) takes ~11 min on a decent machine;
   training-set extraction (900k points) takes ~1 min.
5. Outputs open in CloudCompare: select the cloud in the DB Tree, set the active scalar
   field to "Classification" to color by predicted class.
