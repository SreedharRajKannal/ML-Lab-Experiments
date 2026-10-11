import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("pima.csv")

X = df.drop(columns = ["diabetic"])
y = df["diabetic"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

unscaled_model = LogisticRegression(max_iter=1000)

unscaled_model.fit(x_train,y_train)

y_unscaled = unscaled_model.predict(x_test)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

scaled_model = LogisticRegression(max_iter=1000)

scaled_model.fit(x_train_scaled,y_train)

y_scaled = scaled_model.predict(x_test_scaled)

def performance(model, y,y_pred):
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"--- {model} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")
performance("Unscaled Logistic Regression", y_test, y_unscaled)
performance("Scaled Logistic Regression", y_test, y_scaled)