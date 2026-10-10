import pandas as pd

data ={"Name" :["Spongbob","Patric","Squadward"],
       "Age":[20,30,50]}
df = pd.DataFrame(data)


#adding a new column
df["Job"] =["cook","N/A","Cashier"]


print(df.iloc[1])