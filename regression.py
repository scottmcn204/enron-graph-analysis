import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

features = sorted(glob.glob("features*.csv"))[:17]
dfs = [pd.read_csv(f) for f in features]
data = pd.concat(dfs, ignore_index=True)

X = data[['adamic_adar', 'jaccard', 'preferential_attachment']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))