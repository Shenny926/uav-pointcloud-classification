import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---- Load features ----
data_path = r"E:\shenj\Documents\pointcloud_2024_2025\training_features.npz"
data = np.load(data_path, allow_pickle=True)

features = data["features"]
labels = data["labels"]
feature_names = list(data["feature_names"])

label_names = {1: "building", 2: "road", 3: "vegetation"}
target_names = [label_names[l] for l in sorted(label_names)]

print("Feature matrix:", features.shape)
print("Labels:", labels.shape)

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ---- Train Random Forest ----
print("\nTraining Random Forest...")
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)

# ---- Evaluate ----
print("\nEvaluating on held-out test set...")
y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=target_names))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print("Rows = true label, Columns = predicted label")
print("Order:", target_names)
print(cm)

# ---- Feature importance ----
print("\n=== Feature Importance ===")
importances = clf.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# ---- Save trained model ----
model_path = r"E:\shenj\Documents\pointcloud_2024_2025\rf_classifier.joblib"
joblib.dump(clf, model_path)
print(f"\nSaved trained model to: {model_path}")