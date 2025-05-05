import pandas as pd

employee = pd.read_csv("EnronEmployeeInformation_with_roles.csv")
employee.to_csv("enron_info", index=False, quoting=1)

