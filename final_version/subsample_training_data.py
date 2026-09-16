import numpy as np

# ---- Load merged training data ----
data_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_data.npz"
data = np.load(data_path)

xyz = data["xyz"]
labels = data["labels"]
intensity = data["intensity"]
rgb = data["rgb"]

print("Loaded total points:", len(labels))

# ---- Subsample settings ----
MAX_PER_CLASS = 300_000
rng = np.random.default_rng(seed=42)  # fixed seed = reproducible results

label_names = {1: "building", 2: "road", 3: "vegetation"}

sampled_idx = []

for label_val, name in label_names.items():
    class_idx = np.where(labels == label_val)[0]
    n_available = len(class_idx)
    n_take = min(MAX_PER_CLASS, n_available)
    chosen = rng.choice(class_idx, size=n_take, replace=False)
    sampled_idx.append(chosen)
    print(f"{name}: sampled {n_take} of {n_available}")

sampled_idx = np.concatenate(sampled_idx)
rng.shuffle(sampled_idx)  # mix classes together

# ---- Apply sampling ----
xyz_s = xyz[sampled_idx]
labels_s = labels[sampled_idx]
intensity_s = intensity[sampled_idx]
rgb_s = rgb[sampled_idx]

print("\nSubsampled total points:", len(labels_s))

# ---- Save ----
output_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_data_subsampled.npz"
np.savez(output_path, xyz=xyz_s, labels=labels_s, intensity=intensity_s, rgb=rgb_s)
print(f"Saved subsampled training data to: {output_path}")