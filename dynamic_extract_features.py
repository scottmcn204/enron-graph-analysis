import glob
import pandas as pd

features = sorted(glob.glob("features/features*.csv"))[:17]
feature_frames = [pd.read_csv(f) for f in features]

window_size = 3
lookahead = 1 
data_sequences = []
X = []
y = []
edges = []
target_frame = feature_frames[-1]
feature_frames = feature_frames[:-1]

# Loop through edges in the target month
for idx, row in target_frame.iterrows():
    u, v = row['u'], row['v']
    label = row['label']
    edge_features = []

    for frame in feature_frames:
        match = frame[(frame['u'] == u) & (frame['v'] == v)]
        if not match.empty:
            edge_features.extend([
                match['adamic_adar'].values[0],
                match['jaccard'].values[0],
                match['common_neighbors'].values[0]
            ])
        else:
            # Edge not present in this month
            edge_features.extend(["err", "err", "err"])

    X.append(edge_features)
    y.append(label)
    edges.append((u, v))  # Keep track of the edge

# Convert to DataFrame if needed
X_df = pd.DataFrame(X)
y_series = pd.Series(y)

print(X_df)