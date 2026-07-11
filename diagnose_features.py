import numpy as np
import laspy

# ---- Load training features (has greenness + z per class) ----
train_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_features.npz"
train_data = np.load(train_path, allow_pickle=True)

features = train_data["features"]
labels = train_data["labels"]
feature_names = list(train_data["feature_names"])

z_idx = feature_names.index("z")
green_idx = feature_names.index("greenness")

label_names = {1: "building", 2: "road", 3: "vegetation"}

print("=== TRAINING DATA (2024+2025) ===")
for val, name in label_names.items():
    mask = labels == val
    z_vals = features[mask, z_idx]
    g_vals = features[mask, green_idx]
    print(f"\n{name}:")
    print(f"  z:         mean={z_vals.mean():.2f}  std={z_vals.std():.2f}  min={z_vals.min():.2f}  max={z_vals.max():.2f}")
    print(f"  greenness: mean={g_vals.mean():.4f}  std={g_vals.std():.4f}  min={g_vals.min():.4f}  max={g_vals.max():.4f}")

# ---- Load 2026 classified point cloud (already has classification + xyz + rgb) ----
classified_path = r"E:\shenj\Documents\test_set_2026\PhotoLab_Group1_2026_classified.laz"
las2026 = laspy.read(classified_path)

z2026 = np.array(las2026.z)
r = np.array(las2026.red, dtype=np.float64)
g = np.array(las2026.green, dtype=np.float64)
b = np.array(las2026.blue, dtype=np.float64)
sum_rgb = r + g + b + 1e-6
greenness2026 = g / sum_rgb
predicted_labels = np.array(las2026.classification)

print("\n\n=== 2026 DATA (predicted classes) ===")
for val, name in label_names.items():
    mask = predicted_labels == val
    if mask.sum() == 0:
        print(f"\n{name}: no points predicted")
        continue
    z_vals = z2026[mask]
    g_vals = greenness2026[mask]
    print(f"\n{name} (n={mask.sum()}):")
    print(f"  z:         mean={z_vals.mean():.2f}  std={z_vals.std():.2f}  min={z_vals.min():.2f}  max={z_vals.max():.2f}")
    print(f"  greenness: mean={g_vals.mean():.4f}  std={g_vals.std():.4f}  min={g_vals.min():.4f}  max={g_vals.max():.4f}")

print("\n\n=== OVERALL RGB / GREENNESS COMPARISON ===")
print("2026 overall greenness: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
    greenness2026.mean(), greenness2026.std(), greenness2026.min(), greenness2026.max()
))

train_green_all = features[:, green_idx]
print("Training overall greenness: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
    train_green_all.mean(), train_green_all.std(), train_green_all.min(), train_green_all.max()
))

print("\n2026 overall z: mean={:.2f} std={:.2f} min={:.2f} max={:.2f}".format(
    z2026.mean(), z2026.std(), z2026.min(), z2026.max()
))
train_z_all = features[:, z_idx]
print("Training overall z: mean={:.2f} std={:.2f} min={:.2f} max={:.2f}".format(
    train_z_all.mean(), train_z_all.std(), train_z_all.min(), train_z_all.max()
))