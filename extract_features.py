import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
 
# ---- Load subsampled training data ----
data_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_data_subsampled.npz"
data = np.load(data_path)
 
xyz = data["xyz"]
labels = data["labels"]
intensity = data["intensity"]
rgb = data["rgb"]
 
n_points = len(xyz)
print("Points to process:", n_points)
 
# ---- Settings ----
K = 20  # number of neighbors to use for local geometry
 
# ---- Step 1: Find K nearest neighbors for every point ----
print("Building neighbor search structure...")
t0 = time.time()
nn = NearestNeighbors(n_neighbors=K, algorithm="kd_tree", n_jobs=-1)
nn.fit(xyz)
print(f"  done in {time.time()-t0:.1f}s")
 
print("Querying neighbors for all points (this is the slow part)...")
t0 = time.time()
distances, indices = nn.kneighbors(xyz)
print(f"  done in {time.time()-t0:.1f}s")
 
# ---- Step 2: Compute geometric features via local PCA ----
print("Computing geometric features...")
t0 = time.time()
 
linearity = np.zeros(n_points)
planarity = np.zeros(n_points)
sphericity = np.zeros(n_points)
verticality = np.zeros(n_points)
roughness = np.zeros(n_points)
height_range = np.zeros(n_points)
 
for i in range(n_points):
    neighbor_pts = xyz[indices[i]]  # K x 3
    centroid = neighbor_pts.mean(axis=0)
    centered = neighbor_pts - centroid
 
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)  # avoid div by zero
    l1, l2, l3 = eigvals[2], eigvals[1], eigvals[0]  # l1 >= l2 >= l3
 
    linearity[i] = (l1 - l2) / l1
    planarity[i] = (l2 - l3) / l1
    sphericity[i] = l3 / l1
 
    # Verticality: how aligned the smallest-eigenvalue eigenvector is with Z axis
    normal = eigvecs[:, 0]  # eigenvector for smallest eigenvalue
    verticality[i] = 1.0 - abs(normal[2])  # 0 = flat/horizontal, 1 = vertical
 
    # Roughness: std of distances of neighbors to local best-fit plane (approx via smallest eigval)
    roughness[i] = np.sqrt(l3)
 
    # Local height range within neighborhood
    height_range[i] = neighbor_pts[:, 2].max() - neighbor_pts[:, 2].min()
 
    if i % 100000 == 0:
        print(f"  {i}/{n_points}")
 
print(f"  done in {time.time()-t0:.1f}s")
 
# ---- Step 3: Assemble full feature matrix ----
z = xyz[:, 2]
 
# Simple greenness index from RGB (helps separate vegetation)
r = rgb[:, 0].astype(np.float32)
g = rgb[:, 1].astype(np.float32)
b = rgb[:, 2].astype(np.float32)
sum_rgb = r + g + b + 1e-6
greenness = g / sum_rgb
 
features = np.column_stack([
    z,
    intensity,
    linearity,
    planarity,
    sphericity,
    verticality,
    roughness,
    height_range,
    greenness,
])
 
feature_names = [
    "z", "intensity", "linearity", "planarity",
    "sphericity", "verticality", "roughness", "height_range", "greenness"
]
 
print("\nFeature matrix shape:", features.shape)
print("Feature names:", feature_names)
 
# ---- Save ----
output_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_features.npz"
np.savez(output_path, features=features, labels=labels, feature_names=feature_names)
print(f"\nSaved features to: {output_path}")
 