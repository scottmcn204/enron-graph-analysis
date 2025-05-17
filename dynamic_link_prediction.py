import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

train_months = list(range(0, 14))
test_months = list(range(14, 17))
train_files = [f"features/features{m}.csv" for m in train_months]
test_files = [f"features/features{m}.csv" for m in test_months]
train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
actual_edges = test_df[test_df['label'] == 1]['edge'].to_list()
feature_cols = [ 'adamic_adar', 'jaccard', 'common_neighbors', 'edge_persistence', 'cc_u_delta', 'cc_v_delta']
X_train = train_df[feature_cols]
y_train = train_df['label']
X_test = test_df[feature_cols]
y_test = test_df['label']
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
edges = test_df['edge']
predicted_edges = [edge for edge, pred in zip(edges, y_pred) if pred == 1]


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

with open("predicted_edges.pkl", "wb") as f:
    pickle.dump(predicted_edges, f)

with open("actual_edges.pkl", "wb") as f:
    pickle.dump(actual_edges, f)

y_scores = model.predict_proba(X_test)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')  # random classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Dynamic Link Prediction")
plt.legend()
plt.grid(True)
plt.savefig("auc.png")

importances = model.feature_importances_
feature_names = X_train.columns
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("plots/dynamicimportances.png")