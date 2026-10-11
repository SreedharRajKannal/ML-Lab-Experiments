# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load and Preprocess the Dataset
# The syllabus specifies using the California Housing dataset
california = fetch_california_housing()
X = california.data[:, 2].reshape(-1, 1) # Using a single feature (e.g., average rooms)
y = california.target

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Implement Linear Regression
# Normal Equation method (Standard Scikit-Learn implementation)[cite: 1]
model_normal = LinearRegression()
model_normal.fit(X_train, y_train)

# Gradient Descent method (Stochastic Gradient Descent)[cite: 1]
model_gd = SGDRegressor(max_iter=1000, tol=1e-3)
model_gd.fit(X_train, y_train)

# 4. Predict and Evaluate[cite: 1]
y_pred = model_normal.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# 5. Visualize the fitted line[cite: 1]
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Average Rooms')
plt.ylabel('House Price')
plt.title('Linear Regression: California Housing')
plt.legend()
plt.show()
