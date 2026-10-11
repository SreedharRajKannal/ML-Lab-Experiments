import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

print("Fetching dataset from OpenML")
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

X = X[:5000]
y = y[:5000]

x_scaled = X/255.0

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)

k = ['linear','poly','rbf']
for i in k:
    start = time.time()
    model = SVC(kernel=i,random_state=42)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    end = time.time()
    total = end - start
    
    acc= accuracy_score(y_test,y_pred)
    print(f"Accuracy for model {i} is {acc:.2f}")
