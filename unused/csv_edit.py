import pandas as pd

employees = pd.read_csv("enron_info.csv")
employees['Name'] = employees['Name'].astype(str).str.lower().str.replace(' ', '',regex=False).str.replace('.','',regex=False).str.replace('-', '')
employees.to_csv("enron_info2.csv", index=False, quoting=1)

