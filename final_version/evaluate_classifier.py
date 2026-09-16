"""Evaluate a classified cloud against the three labelled 2026 LAZ files."""
from pathlib import Path
import argparse

import laspy
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
REFERENCE_DIR = WORKSPACE / "extracted_test" / "test_set_2026"
LABELS = {1: "building", 2: "road", 3: "vegetation"}


def first_existing(*paths):
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def defaults():
    old_root = Path(r"C:\Users\nicho\PycharmProjects\PythonProject\Project")
    classified = first_existing(
        SCRIPT_DIR / "outputs" / "PhotoLab_Group1_2026_classified_v3.laz",
        old_root / "outputs" / "PhotoLab_Group1_2026_classified_v3.laz",
    )
    return classified, {
        1: first_existing(
            REFERENCE_DIR / "PhotoLab_Group1-dense_point_cloud_Test_set.2026_buildings.laz",
            old_root / "data" / "2026" / "PhotoLab_Group1-dense_point_cloud_Test_set.2026_buildings.laz",
        ),
        2: first_existing(
            REFERENCE_DIR / "PhotoLab_Group1-dense_point_cloud_Test_set_2026_roads.laz",
            old_root / "data" / "2026" / "PhotoLab_Group1-dense_point_cloud_Test_set_2026_roads.laz",
        ),
        3: first_existing(
            REFERENCE_DIR / "PhotoLab_Group1-dense_point_cloud_Test_set_2026_vegetation.laz",
            old_root / "data" / "2026" / "PhotoLab_Group1-dense_point_cloud_Test_set_2026_vegetation.laz",
        ),
    }


def coordinate_hash(x, y, z):
    """Fast 64-bit hash; coordinate equality is verified after lookup."""
    x = np.asarray(x, dtype=np.int64).view(np.uint64)
    y = np.asarray(y, dtype=np.int64).view(np.uint64)
    z = np.asarray(z, dtype=np.int64).view(np.uint64)
    return (x * np.uint64(0x9E3779B185EBCA87)) ^ (
        y * np.uint64(0xC2B2AE3D27D4EB4F)
    ) ^ (z * np.uint64(0x165667B19E3779F9))


def same_grid(header_a, header_b):
    return np.array_equal(header_a.scales, header_b.scales) and np.array_equal(
        header_a.offsets, header_b.offsets
    )


def parse_args():
    classified, refs = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified", type=Path, default=classified)
    parser.add_argument("--building", type=Path, default=refs[1])
    parser.add_argument("--road", type=Path, default=refs[2])
    parser.add_argument("--vegetation", type=Path, default=refs[3])
    return parser.parse_args()


def main():
    args = parse_args()
    references = {1: args.building, 2: args.road, 3: args.vegetation}
    for path in [args.classified, *references.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Required point cloud not found: {path}")

    print(f"Loading classified cloud:\n  {args.classified}")
    classified = laspy.read(args.classified)
    predicted = np.asarray(classified.classification, dtype=np.uint8)
    full_hash = coordinate_hash(classified.X, classified.Y, classified.Z)
    print(f"Indexing {len(predicted):,} classified points...")
    order = np.argsort(full_hash)
    sorted_hash = full_hash[order]

    duplicate_hashes = int(np.count_nonzero(sorted_hash[1:] == sorted_hash[:-1]))
    if duplicate_hashes:
        print(f"Warning: {duplicate_hashes:,} duplicate coordinate hashes; ambiguous duplicates will be skipped.")

    true_parts, predicted_parts = [], []
    missing = ambiguous = 0
    for label, path in references.items():
        print(f"Matching {LABELS[label]} reference: {path.name}")
        reference = laspy.read(path)
        if not same_grid(classified.header, reference.header):
            raise ValueError(f"Coordinate scale/offset differs for {path}; exact matching is unsafe.")
        ref_hash = coordinate_hash(reference.X, reference.Y, reference.Z)
        left = np.searchsorted(sorted_hash, ref_hash, side="left")
        right = np.searchsorted(sorted_hash, ref_hash, side="right")
        valid = (left < len(sorted_hash)) & (right == left + 1)
        ambiguous += int(np.count_nonzero(right > left + 1))
        candidates = np.zeros(len(reference.points), dtype=np.int64)
        candidates[valid] = order[left[valid]]
        exact = valid.copy()
        idx = candidates[valid]
        exact[valid] = (
            (np.asarray(classified.X)[idx] == np.asarray(reference.X)[valid])
            & (np.asarray(classified.Y)[idx] == np.asarray(reference.Y)[valid])
            & (np.asarray(classified.Z)[idx] == np.asarray(reference.Z)[valid])
        )
        missing += int(np.count_nonzero(~exact))
        true_parts.append(np.full(np.count_nonzero(exact), label, dtype=np.uint8))
        predicted_parts.append(predicted[candidates[exact]])

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(predicted_parts)
    print(f"\nMatched points: {len(y_true):,} / {len(predicted):,}")
    if missing or ambiguous:
        print(f"Unmatched: {missing:,}; ambiguous duplicate matches: {ambiguous:,}")
    if not len(y_true):
        raise RuntimeError("No reference points could be matched")

    label_values = list(LABELS)
    target_names = [LABELS[x] for x in label_values]
    print(f"\nOverall accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print("\n=== Per-class metrics ===")
    print(classification_report(
        y_true, y_pred, labels=label_values, target_names=target_names,
        digits=4, zero_division=0,
    ))
    print("=== Confusion matrix ===")
    print("Rows = true class; columns = predicted class")
    print("Order:", target_names)
    print(confusion_matrix(y_true, y_pred, labels=label_values))


if __name__ == "__main__":
    main()
