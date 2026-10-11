import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns = california.feature_names)

df["price"] = california.target

X=df[["AveRooms"]]
y=df["price"]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
GD = SGDRegressor(max_iter = 1000)

linear_model.fit(x_train,y_train)
GD.fit(x_train,y_train)

y_linear = linear_model.predict(x_test)
y_gd = GD.predict(x_test)

def errors(a):
    mse = mean_squared_error(y_test,a)
    r2 = r2_score(y_test,a)
    print(f"MSE : {mse:.2f}")
    print(f"R2 : {r2:.2f}")
print("Linear Regression:")
errors(y_linear)
print("GD:")
errors(y_gd)

plt.title("Linear VS GD")
plt.scatter(x_test,y_test,marker=".",color = "Blue",label = "Actual data")
plt.plot(x_test,y_linear,label="Linear Regression")
plt.plot(x_test,y_gd, label = "GD")
plt.xlabel("AVRooms")
plt.ylabel("Price")
plt.legend()
plt.show()
