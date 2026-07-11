import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

# ---- Load original training features ----
data_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_features.npz"
data = np.load(data_path, allow_pickle=True)

features = data["features"].copy()
labels = data["labels"]
feature_names = list(data["feature_names"])

green_idx = feature_names.index("greenness")

# ---- Normalize greenness using THIS dataset's own stats ----
train_green_mean = features[:, green_idx].mean()
train_green_std = features[:, green_idx].std()

print(f"Training greenness stats: mean={train_green_mean:.4f}, std={train_green_std:.4f}")

features[:, green_idx] = (features[:, green_idx] - train_green_mean) / train_green_std

# ---- Save normalization stats so we can apply the SAME logic (per-dataset) to 2026 ----
norm_stats_path = r"E:\shenj\Documents\pointcloud_2024_2025\greenness_norm_stats.json"
with open(norm_stats_path, "w") as f:
    json.dump({"train_green_mean": train_green_mean, "train_green_std": train_green_std}, f)
print(f"Saved normalization stats to: {norm_stats_path}")

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ---- Train Random Forest ----
print("\nTraining Random Forest (normalized greenness)...")
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)

# ---- Evaluate ----
label_names = {1: "building", 2: "road", 3: "vegetation"}
target_names = [label_names[l] for l in sorted(label_names)]

y_pred = clf.predict(X_test)

print("\n=== Classification Report (normalized greenness) ===")
print(classification_report(y_test, y_pred, target_names=target_names))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print("Order:", target_names)
print(cm)

print("\n=== Feature Importance ===")
importances = clf.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# ---- Save trained model ----
model_path = r"E:\shenj\Documents\pointcloud_2024_2025\rf_classifier_normalized.joblib"
joblib.dump(clf, model_path)
print(f"\nSaved trained model to: {model_path}")