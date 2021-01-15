import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


column_transformer = ColumnTransformer(transformers=[('encoder', 
                                                       OneHotEncoder(),
                                                       [3])],
                                       remainder='passthrough')
X = np.array(column_transformer.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)

y_comparison = np.concatenate((y_test.reshape(len(y_test), 1),
                               y_prediction.reshape(len(y_prediction), 1)),
                              axis=1)
