import laspy

file_path = r"E:\shenj\Documents\test_set_2026\PhotoLab_Group1-dense_point_cloud_Test_set.2026.laz"

las = laspy.read(file_path)

print("Number of points:", len(las.points))
print("Point format:", las.header.point_format)
print("Available dimensions:", list(las.point_format.dimension_names))
print("Classification values present:", set(las.classification))