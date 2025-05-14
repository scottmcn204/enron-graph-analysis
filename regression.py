import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

features = sorted(glob.glob("features/features*.csv"))[:17]
dfs = [pd.read_csv(f) for f in features]
data = pd.concat(dfs, ignore_index=True)

edges = data['edge']
X = data[['adamic_adar', 'jaccard', 'common_neighbors']]
y = data['label']

X_train, X_test, y_train, y_test, edges_train, edges_test = train_test_split(
    X, y, edges, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

predicted_edges = [eval(edge) for edge, label in zip(edges_test, y_pred) if label == 1]

with open("predicted_edges.pkl", "wb") as f:
    pickle.dump(predicted_edges, f)



corr = data.drop(columns=["edge"]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.savefig("plots/featurecorr.png")
