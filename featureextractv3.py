import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

# ---- Load subsampled training data (has raw xyz, intensity, rgb) ----
data_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_data_subsampled.npz"
data = np.load(data_path)

xyz = data["xyz"]
labels = data["labels"]
intensity = data["intensity"]
rgb = data["rgb"]

n_points = len(xyz)
print("Points to process:", n_points)

K = 20

# ---- Nearest neighbors ----
print("Building neighbor search structure...")
t0 = time.time()
nn = NearestNeighbors(n_neighbors=K, algorithm="kd_tree", n_jobs=-1)
nn.fit(xyz)
print(f"  done in {time.time()-t0:.1f}s")

print("Querying neighbors for all points...")
t0 = time.time()
distances, indices = nn.kneighbors(xyz)
print(f"  done in {time.time()-t0:.1f}s")

# ---- Precompute raw greenness for ALL points (needed for local averaging) ----
r = rgb[:, 0].astype(np.float64)
g = rgb[:, 1].astype(np.float64)
b = rgb[:, 2].astype(np.float64)
sum_rgb = r + g + b + 1e-6
greenness_raw = g / sum_rgb

# ---- Compute geometric features + LOCAL RELATIVE greenness ----
print("Computing geometric features + local relative greenness...")
t0 = time.time()

linearity = np.zeros(n_points)
planarity = np.zeros(n_points)
sphericity = np.zeros(n_points)
verticality = np.zeros(n_points)
roughness = np.zeros(n_points)
height_range = np.zeros(n_points)
local_greenness = np.zeros(n_points)  # point's greenness minus its neighborhood's mean greenness

for i in range(n_points):
    neighbor_idx = indices[i]
    neighbor_pts = xyz[neighbor_idx]
    centroid = neighbor_pts.mean(axis=0)
    centered = neighbor_pts - centroid

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    l1, l2, l3 = eigvals[2], eigvals[1], eigvals[0]

    linearity[i] = (l1 - l2) / l1
    planarity[i] = (l2 - l3) / l1
    sphericity[i] = l3 / l1

    normal = eigvecs[:, 0]
    verticality[i] = 1.0 - abs(normal[2])

    roughness[i] = np.sqrt(l3)
    height_range[i] = neighbor_pts[:, 2].max() - neighbor_pts[:, 2].min()

    # Local relative greenness: how much greener/less green is this point vs its local neighborhood
    local_mean_green = greenness_raw[neighbor_idx].mean()
    local_greenness[i] = greenness_raw[i] - local_mean_green

    if i % 100000 == 0:
        print(f"  {i}/{n_points}")

print(f"  done in {time.time()-t0:.1f}s")

# ---- Assemble feature matrix ----
z = xyz[:, 2]

features = np.column_stack([
    z,
    intensity,
    linearity,
    planarity,
    sphericity,
    verticality,
    roughness,
    height_range,
    local_greenness,   # replaces global greenness
])

feature_names = [
    "z", "intensity", "linearity", "planarity",
    "sphericity", "verticality", "roughness", "height_range", "local_greenness"
]

print("\nFeature matrix shape:", features.shape)
print("Feature names:", feature_names)

# ---- Save ----
output_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_features_v3.npz"
np.savez(output_path, features=features, labels=labels, feature_names=feature_names)
print(f"\nSaved features to: {output_path}")