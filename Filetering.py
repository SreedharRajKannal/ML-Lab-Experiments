import pandas as pd

df = pd.read_csv("FirstGenPokemon.csv",index_col="Name")
print(df.head())
tall_pokemon = df[df["Height(m)"]>2]


print(tall_pokemon)


Heavy_pokemon = df[df["Weight(kg)"]>100]

print(Heavy_pokemon)


print(df.min(numeric_only=True))