import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

print("Fetching dataset from OpenML")
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

X = X[:1000]
y = y[:1000]

x_scaled = X/255

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)

neighbour = [1,3,5,9]
for k in neighbour:
    knn = KNeighborsClassifier(n_neighbors=k)
    start_time =time.time()
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    end_time = time.time()
    total_time =end_time-start_time
    acc =accuracy_score(y_test,y_pred)
    print(f"accuracy with {k} Neighbours ={acc}% Time Taken ={total_time:.2f}s")

