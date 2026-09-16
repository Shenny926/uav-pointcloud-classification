from pathlib import Path
import argparse
import time

import joblib
import laspy
import numpy as np

from featureextractv3 import FEATURE_NAMES, extract_features
from spatial_filter import voxel_majority

LABEL_NAMES = {1: "building", 2: "road", 3: "vegetation"}
SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_INPUT = Path(
    r"C:\Users\nicho\PycharmProjects\PythonProject\Project\data\2026\PhotoLab_Group1-dense_point_cloud_Test_set.2026.laz"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="?", default=LEGACY_INPUT,
                        help="Unclassified input .las/.laz")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "outputs" / "rf_classifier_v3.joblib")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "PhotoLab_Group1_2026_classified_v3.laz")
    parser.add_argument("-k", type=int, help="Override the model's neighbour count")
    parser.add_argument("--vegetation-threshold", type=float, help="Override threshold learned on validation data")
    parser.add_argument("--spatial-voxel", type=float, default=0.75,
                        help="3-D majority-filter voxel size in metres; use 0 to disable")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input point cloud not found: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {args.model}\nRun Classifierv3.py first."
        )
    print(f"Loading {args.input}...")
    las = laspy.read(args.input)
    n = len(las.points)
    if n == 0:
        raise ValueError("Input point cloud is empty")
    xyz = np.column_stack((las.x, las.y, las.z))
    intensity = np.asarray(las.intensity, dtype=np.float64)
    dims = set(las.point_format.dimension_names)
    if not {"red", "green", "blue"}.issubset(dims):
        raise ValueError("RGB dimensions are required for reliable vegetation classification")
    rgb = np.column_stack((las.red, las.green, las.blue))

    saved = joblib.load(args.model)
    if not isinstance(saved, dict) or "model" not in saved:
        raise ValueError("Old v3 model detected. Re-run featureextractv3.py and Classifierv3.py first.")
    model = saved["model"]
    expected = list(saved.get("feature_names", []))
    if expected != FEATURE_NAMES:
        raise ValueError(f"Model feature mismatch. Expected {FEATURE_NAMES}, model has {expected}")

    neighbor_count = args.k if args.k is not None else int(saved.get("neighbor_count", 30))
    print(f"Extracting {len(FEATURE_NAMES)} features for {n:,} points (k={neighbor_count})...")
    t0 = time.time()
    features = extract_features(xyz, intensity, rgb, neighbor_count)
    print(f"Feature extraction completed in {time.time()-t0:.1f}s")

    probabilities = model.predict_proba(features)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    threshold = args.vegetation_threshold
    if threshold is None:
        threshold = float(saved.get("vegetation_threshold", 0.40))
    veg_columns = np.flatnonzero(model.classes_ == 3)
    if len(veg_columns):
        predictions[probabilities[:, veg_columns[0]] >= threshold] = 3
    print(f"Vegetation probability threshold: {threshold:.3f}")
    if args.spatial_voxel > 0:
        print(f"Applying {args.spatial_voxel:.2f} m spatial consistency filter...")
        predictions = voxel_majority(
            las.X, las.Y, las.Z, predictions, args.spatial_voxel, las.header.scales
        )

    print("\n=== Prediction summary (improved v3) ===")
    for value, name in LABEL_NAMES.items():
        count = int(np.count_nonzero(predictions == value))
        print(f"  {name}: {count:,} points ({100.0*count/n:.1f}%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    las.classification = predictions.astype(np.uint8)
    las.write(args.output)
    print(f"\nSaved classified point cloud to: {args.output}")


if __name__ == "__main__":
    main()
