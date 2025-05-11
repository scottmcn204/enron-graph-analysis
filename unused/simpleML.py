from sklearn.model_selection import train_test_split
import pandas as pd

features = pd.read_csv("graph_features.csv")
roles = pd.read_csv("enron_info2,csv")

X = list(features)