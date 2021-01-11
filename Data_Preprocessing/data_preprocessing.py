import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

standard_scaler = StandardScaler()
X_train[:, 3:] = standard_scaler.fit_transform(X_train[:, 3:])
X_test[:, 3:] = standard_scaler.transform(X_test[:, 3:])
