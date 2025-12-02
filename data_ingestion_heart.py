import pandas as pd

data=pd.read_csv("heart (1).csv")
data=data.dropna()
data=data.reset_index()
print(data.head())