import pandas as pd

df = pd.read_csv("FirstGenPokemon.csv",index_col="Name")

df.drop(columns=["Number"])
print(df)