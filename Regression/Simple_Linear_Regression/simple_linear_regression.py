import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_test_prediction = regressor.predict(X_test)

y_train_prediction = regressor.predict(X_train)
plt.scatter(X_train, y_train, color='black')
plt.plot(X_train, y_train_prediction, color='blue')
plt.ylabel('Salary')
plt.xlabel('Experience(Years)')
plt.title('Experience X Salary\nTraining Set')
plt.show()

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_test_prediction, color='red')
plt.ylabel('Salary')
plt.xlabel('Experience(Years)')
plt.title('Experience X Salary\nTest Set')
plt.show()
