#pgm2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("auto-mpg.csv")
df = df.dropna()


X = df[["displacement"]]
y = df["mpg"]
#Splitting
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

#Models
linear = LinearRegression()
poly2 = PolynomialFeatures(degree=2)
poly3 = PolynomialFeatures(degree=3)
poly4 = PolynomialFeatures(degree=4)
poly5 = PolynomialFeatures(degree=5)

model_poly2 = LinearRegression()
model_poly3 = LinearRegression()
model_poly4 = LinearRegression()
model_poly5 = LinearRegression()

#Train
linear.fit(x_train,y_train)

x_train_poly2 = poly2.fit_transform(x_train)
x_test_poly2 = poly2.transform(x_test)

x_train_poly3 = poly3.fit_transform(x_train)
x_test_poly3 = poly3.transform(x_test)

x_train_poly4 = poly4.fit_transform(x_train)
x_test_poly4 = poly4.transform(x_test)

x_train_poly5 = poly5.fit_transform(x_train)
x_test_poly5 = poly5.transform(x_test)

model_poly2.fit(x_train_poly2,y_train)
model_poly3.fit(x_train_poly3,y_train)
model_poly4.fit(x_train_poly4,y_train)
model_poly5.fit(x_train_poly5,y_train)


#errors
def error(a) :
    mse = mean_squared_error(y_test,a)
    r2 = r2_score(y_test,a)
    print(f"MSE:{mse:.2f}\nR2:{r2:.2f}")

#Prediction
y_linear = linear.predict(x_test)
y_poly2 = model_poly2.predict(x_test_poly2)
y_poly3 = model_poly3.predict(x_test_poly3)
y_poly4 = model_poly4.predict(x_test_poly4)
y_poly5 = model_poly5.predict(x_test_poly5)

L = [y_poly2,y_poly3,y_poly4,y_poly5]
count = 2
print("Linear Regression")
error (y_linear)
for i in L:
    print("Polynomial Regression with degree ",count)
    count+=1
    error (i)

x_plot = np.linspace(X.min(),X.max(),100).reshape(-1,1)

y_plot_linear = linear.predict(x_plot)
y_plot_poly2 = model_poly2.predict(poly2.transform(x_plot))
y_plot_poly3 = model_poly3.predict(poly3.transform(x_plot))
y_plot_poly4 = model_poly4.predict(poly4.transform(x_plot))
y_plot_poly5 = model_poly5.predict(poly5.transform(x_plot))

plt.title("Linear Vs Polynomial")
plt.scatter(x_test, y_test, color="gray", alpha=0.5, label="Actual Data")

plt.plot(x_plot, y_plot_linear, color="blue", label="Linear", linestyle="dashed")
plt.plot(x_plot, y_plot_poly2, color="red", label="deg2")
plt.plot(x_plot, y_plot_poly3, color="yellow", label="deg3")
plt.plot(x_plot, y_plot_poly4, color="cyan", label="deg4")
plt.plot(x_plot, y_plot_poly5, color="green", label="deg5")

plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.legend()
plt.show()
