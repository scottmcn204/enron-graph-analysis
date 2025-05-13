import networkx as nx
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

emails = pd.read_csv("emails.csv")
edge_weights = defaultdict(int)
monthly_graphs = list()

# pd.set_option('display.max_colwidth', None)
# print(emails['message'].head())

def extract_sender(message):
    match = re.search(r'From:\s(.+)', str(message))
    if match:
        senders = match.group(1)
        senders = senders.split('<')[0]
        return [r.strip() for r in senders.split(',')]
    else:
        return []

def extract_recipients(message):
    match = re.search(r'To:\s(.+)', str(message))
    if match:
        recipients = match.group(1)
        recipients = recipients.split('<')[0]
        return [r.strip() for r in recipients.split(',')]
    else:
        return []
    
def extract_date(message):
    match = re.search(r'Date:\s(.+)', str(message))
    if match:
        return pd.to_datetime(match.group(1), errors='coerce')
    else:
        return pd.NaT

emails['sender'] = emails['message'].apply(extract_sender)
emails['recipients'] = emails['message'].apply(extract_recipients)
date_series = emails['message'].str.extract(r'Date:\s(.+)', expand=False) 
clean_date_series = date_series.str.replace(r'\s*\(.*\)', '', regex=True)
emails['date'] = pd.to_datetime(clean_date_series, errors='coerce', format="%a, %d %b %Y %H:%M:%S %z", utc=True)
emails.dropna(subset=['date'], inplace=True)    
emails = emails[(emails['date'] >= '2000-08-01') & (emails['date'] < '2002-02-01')]
emails['month'] = emails['date'].dt.to_period('M')

for index, (period, emailgroup) in enumerate(emails.groupby('month')):
    G = nx.DiGraph()
    edge_weights.clear()
    for _, row in emailgroup.iterrows():
        sender = row['sender'][0]
        recipients = row['recipients']
        for recipient in recipients:
            if sender and recipient and (sender != recipient):
                edge_weights[(sender,recipient)] += 1
    for (sender, recipient), weigth in edge_weights.items():
        senderID = sender.split('@')[0].lower().replace(" ", "").replace("\'", "").replace("\"", "").replace(".", "").replace("_", "")
        recipientID = recipient.split('@')[0].lower().replace(" ", "").replace("\'", "").replace("\"", "").replace(".", "").replace("_", "")
        G.add_edge(senderID, recipientID, weigth=weigth)
    monthly_graphs.append(G)

print(type(monthly_graphs))
print(type(monthly_graphs[0]))
with open("graph.pkl", "wb") as f:
    pickle.dump(monthly_graphs, f)

 
