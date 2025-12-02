import pandas as pd

data=pd.read_csv("publisher/data/weather.csv")
data=data.dropna()
data=data.reset_index()
print(data.head())
print(data.columns)
print(data.iloc[0])