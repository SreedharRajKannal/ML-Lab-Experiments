import pandas as pd

df = pd.read_csv("FirstGenPokemon.csv",index_col="Name")

#first 5
print(df.head())
print("")
#last 5
print(df.tail())
'''
#Full Dataset
print(df.to_string())
'''
print(df.head())
'''
#printing multiple columns

print(df[["Name","Types"]].head())
'''

#Searching for a pokemon from the user input
pk= input("Enter a Pokemon: ")

try:
    print(df.loc[pk].head())
    print(df.loc[pk].tail())
except KeyError:
    print("Pokemon not found")