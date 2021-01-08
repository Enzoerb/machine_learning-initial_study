import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

column_transformer = ColumnTransformer(transformers=[('encoder', 
                                                       OneHotEncoder(),
                                                       [0])],
                                       remainder='passthrough')
X = np.array(column_transformer.fit_transform(X))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
