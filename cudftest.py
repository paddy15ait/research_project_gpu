import pandas as pd


df = pd.read_csv("github_issues.csv")

# np,load not supporeted by numba

print(df.head())

