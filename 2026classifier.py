import numpy as np
import laspy
from sklearn.neighbors import NearestNeighbors
import joblib
import time

# ---- Paths ----
input_path = r"E:\shenj\Documents\test_set_2026\PhotoLab_Group1-dense_point_cloud_Test_set.2026.laz"
model_path = r"E:\shenj\Documents\pointcloud_2024_2025\rf_classifier.joblib"
output_path = r"E:\shenj\Documents\test_set_2026\PhotoLab_Group1_2026_classified.laz"

K = 20  # must match what was used in training

# ---- Load 2026 point cloud ----
print("Loading 2026 point cloud...")
las = laspy.read(input_path)
n_points = len(las.points)
print(f"  -> {n_points} points")

xyz = np.vstack((las.x, las.y, las.z)).T
intensity = np.array(las.intensity, dtype=np.float64)

if hasattr(las, "red"):
    rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float32)
else:
    rgb = np.zeros((n_points, 3), dtype=np.float32)

# ---- Step 1: Nearest neighbors ----
print("\nBuilding neighbor search structure...")
t0 = time.time()
nn = NearestNeighbors(n_neighbors=K, algorithm="kd_tree", n_jobs=-1)
nn.fit(xyz)
print(f"  done in {time.time()-t0:.1f}s")

print("Querying neighbors for all points (slow part)...")
t0 = time.time()
distances, indices = nn.kneighbors(xyz)
print(f"  done in {time.time()-t0:.1f}s")

# ---- Step 2: Geometric features via local PCA ----
print("\nComputing geometric features...")
t0 = time.time()

linearity = np.zeros(n_points)
planarity = np.zeros(n_points)
sphericity = np.zeros(n_points)
verticality = np.zeros(n_points)
roughness = np.zeros(n_points)
height_range = np.zeros(n_points)

for i in range(n_points):
    neighbor_pts = xyz[indices[i]]
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

    if i % 500000 == 0:
        elapsed = time.time() - t0
        print(f"  {i}/{n_points}  ({elapsed:.0f}s elapsed)")

print(f"  done in {time.time()-t0:.1f}s")

# ---- Step 3: Assemble feature matrix (same order as training!) ----
z = xyz[:, 2]
r = rgb[:, 0]
g = rgb[:, 1]
b = rgb[:, 2]
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

print("\nFeature matrix shape:", features.shape)

# ---- Step 4: Load model and predict ----
print("\nLoading trained classifier...")
clf = joblib.load(model_path)

print("Predicting classes for all points...")
t0 = time.time()
predictions = clf.predict(features)
print(f"  done in {time.time()-t0:.1f}s")

label_names = {1: "building", 2: "road", 3: "vegetation"}
print("\n=== Prediction summary ===")
for val, name in label_names.items():
    count = np.sum(predictions == val)
    pct = 100 * count / n_points
    print(f"  {name}: {count} points ({pct:.1f}%)")

# ---- Step 5: Write predictions into classification field and save ----
las.classification = predictions.astype(np.uint8)
las.write(output_path)
print(f"\nSaved classified point cloud to: {output_path}")