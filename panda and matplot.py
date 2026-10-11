import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("FirstGenPokemon.csv")
df.dropna()

type_count = df["Type1"].value_counts()

plt.bar(type_count.index,type_count.values)
plt.show()
