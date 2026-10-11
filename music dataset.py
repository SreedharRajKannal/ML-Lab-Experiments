import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("music.csv")

X = df.drop(columns=["genre"])
y = df["genre"]
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.2,random_state=42)

model = DecisionTreeClassifier()

model.fit(X,y)
predictions = model.predict(x_test)
print(y_test,predictions)
score = accuracy_score(y_test,predictions)
print(score)