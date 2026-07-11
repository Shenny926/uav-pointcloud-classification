import laspy
import numpy as np

# ---- File paths ----
# 2024 files
files_2024 = {
    "building": r"E:\shenj\Documents\pointcloud_2024_2025\2024\Nadir_withRTK-dense_point_cloud_2024_buildings.laz",
    "road": r"E:\shenj\Documents\pointcloud_2024_2025\2024\Nadir_withRTK-dense_point_cloud_2024_roads.laz",
    "vegetation": r"E:\shenj\Documents\pointcloud_2024_2025\2024\Nadir_withRTK-dense_point_cloud_2024_vegetation.laz",
}

# 2025 files
files_2025 = {
    "building": r"E:\shenj\Documents\pointcloud_2024_2025\2025\2025_Nadir-dense_point_cloud_buildings.laz",
    "road": r"E:\shenj\Documents\pointcloud_2024_2025\2025\2025_Nadir-dense_point_cloud_roads.laz",
    "vegetation": r"E:\shenj\Documents\pointcloud_2024_2025\2025\2025_Nadir-dense_point_cloud_vegetation.laz",
}

# ---- Label mapping ----
label_map = {"building": 1, "road": 2, "vegetation": 3}

all_xyz = []
all_labels = []
all_intensity = []
all_rgb = []

def load_and_tag(file_dict, year_label):
    for class_name, path in file_dict.items():
        print(f"Loading {year_label} {class_name}: {path}")
        las = laspy.read(path)
        n = len(las.points)
        print(f"  -> {n} points")

        xyz = np.vstack((las.x, las.y, las.z)).T
        labels = np.full(n, label_map[class_name], dtype=np.uint8)

        all_xyz.append(xyz)
        all_labels.append(labels)

        # Grab intensity and RGB if present
        if hasattr(las, "intensity"):
            all_intensity.append(np.array(las.intensity))
        else:
            all_intensity.append(np.zeros(n))

        if hasattr(las, "red"):
            rgb = np.vstack((las.red, las.green, las.blue)).T
        else:
            rgb = np.zeros((n, 3))
        all_rgb.append(rgb)

load_and_tag(files_2024, "2024")
load_and_tag(files_2025, "2025")

# ---- Merge everything ----
xyz = np.vstack(all_xyz)
labels = np.concatenate(all_labels)
intensity = np.concatenate(all_intensity)
rgb = np.vstack(all_rgb)

print("\n=== Combined training dataset ===")
print("Total points:", len(labels))
for name, val in label_map.items():
    print(f"  {name}: {np.sum(labels == val)} points")

# ---- Save as a single .npz file for later use ----
output_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_data.npz"
np.savez(output_path, xyz=xyz, labels=labels, intensity=intensity, rgb=rgb)
print(f"\nSaved combined training data to: {output_path}")