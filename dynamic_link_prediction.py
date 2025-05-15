import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

train_months = list(range(0, 16))
test_months = list(range(16, 17))
train_files = [f"features/features{m}.csv" for m in train_months]
test_files = [f"features/features{m}.csv" for m in test_months]
train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

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
