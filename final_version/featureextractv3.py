"""Extract robust colour and local-geometry features for v3 training data."""
from pathlib import Path
import argparse
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

FEATURE_NAMES = [
    "z_robust", "intensity_robust", "linearity", "planarity", "sphericity",
    "verticality", "roughness", "height_range", "height_above_local_min",
    "green_ratio", "excess_green", "local_mean_green", "local_greenness",
]
SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_TRAINING_DATA = Path(
    r"C:\Users\nicho\PycharmProjects\PythonProject\Project\outputs\training_data_subsampled.npz"
)


def default_training_data():
    local = SCRIPT_DIR / "outputs" / "training_data_subsampled.npz"
    return local if local.is_file() else LEGACY_TRAINING_DATA


def robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    return (values - median) / max(q75 - q25, 1e-6)


def colour_features(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    total = rgb.sum(axis=1)
    valid = total > 0
    chroma = np.zeros_like(rgb)
    chroma[valid] = rgb[valid] / total[valid, None]
    green_ratio = chroma[:, 1]
    excess_green = 2.0 * chroma[:, 1] - chroma[:, 0] - chroma[:, 2]
    return green_ratio, excess_green


def extract_features(xyz, intensity, rgb, k=30, neighbor_batch_size=100_000):
    n = len(xyz)
    if n < 3:
        raise ValueError("At least three points are required")
    k = min(k, n)
    search = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=-1).fit(xyz)
    green_ratio, excess_green = colour_features(rgb)
    geom = np.zeros((n, 9), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, neighbor_batch_size):
        stop = min(start + neighbor_batch_size, n)
        # Supplying X keeps the query point itself, matching training/inference.
        indices = search.kneighbors(xyz[start:stop], return_distance=False)
        for offset, idx in enumerate(indices):
            i = start + offset
            pts = xyz[idx]
            centered = pts - pts.mean(axis=0)
            eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
            eigvals = np.maximum(eigvals, 1e-12)
            l1, l2, l3 = eigvals[2], eigvals[1], eigvals[0]
            local_green = green_ratio[idx].mean()
            geom[i] = (
                (l1-l2)/l1, (l2-l3)/l1, l3/l1, 1.0-abs(eigvecs[2, 0]),
                np.sqrt(l3), np.ptp(pts[:, 2]), xyz[i, 2]-pts[:, 2].min(),
                local_green, green_ratio[i]-local_green,
            )
        print(f"  {stop:,}/{n:,} ({time.time()-t0:.0f}s)")
    return np.column_stack((
        robust_scale(xyz[:, 2]), robust_scale(intensity), geom[:, :7],
        green_ratio, excess_green, geom[:, 7:]
    )).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_training_data())
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "training_features_v3.npz")
    parser.add_argument("-k", type=int, default=30)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Subsampled training data were not found at:\n  {args.input}\n\n"
            "Pass the correct file with: --input path\\to\\training_data_subsampled.npz"
        )
    data = np.load(args.input)
    features = extract_features(data["xyz"], data["intensity"], data["rgb"], args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, features=features, labels=data["labels"],
        feature_names=FEATURE_NAMES, neighbor_count=np.int32(args.k)
    )
    print(f"Saved {features.shape} feature matrix to: {args.output}")


if __name__ == "__main__":
    main()
