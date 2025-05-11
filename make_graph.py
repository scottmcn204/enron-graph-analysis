import networkx as nx
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

emails = pd.read_csv("emails.csv")
enronInfo = pd.read_csv("enron_info2.csv")
emailAddresses = list(enronInfo['EmailID'])
names = list(enronInfo['Name'])
folders = list(enronInfo['Folder'])
G = nx.DiGraph()
edge_weights = defaultdict(int)

# pd.set_option('display.max_colwidth', None)
# print(emails['message'].head())

def extract_sender(message):
    match = re.search(r'X-From:\s(.+)', str(message))
    if match:
        senders = match.group(1)
        senders = senders.split('<')[0]
        return [r.strip() for r in senders.split(',')]
    else:
        return []

def extract_recipients(message):
    match = re.search(r'X-To:\s(.+)', str(message))
    if match:
        recipients = match.group(1)
        recipients = recipients.split('<')[0]
        return [r.strip() for r in recipients.split(',')]
    else:
        return []

emails['sender'] = emails['message'].apply(extract_sender)
emails['recipients'] = emails['message'].apply(extract_recipients)

for index, row in emails.iterrows():
    sender = row['sender'][0]
    recipients = row['recipients']
    for recipient in recipients:
        if sender and recipient:
            edge_weights[(sender,recipient)] += 1
for (sender, recipient), weigth in edge_weights.items():
    senderID = sender.split('@')[0].lower().replace(" ", "").replace("\'", "").replace("\"", "").replace(".", "").replace("_", "")
    recipientID = recipient.split('@')[0].lower().replace(" ", "").replace("\'", "").replace("\"", "").replace(".", "").replace("_", "")
    if (senderID in names) and (recipientID in names):
        G.add_edge(senderID, recipientID, weigth=weigth)
    
            

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

with open("graph.pkl", "wb") as f:
    pickle.dump(G, f)



with open('allemails.txt', 'w') as output:
    for node in G.nodes:
        output.write(node + "\n")
 
