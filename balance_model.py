from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
# from xgboost import XGBClassifier

dataset = pd.read_csv("./balances.csv")
dataset.head()

for column in dataset.columns:
  dataset[column].fillna(dataset[column].mean(), inplace=True)

X = dataset.drop(['Sum', 'Unnamed: 0'], axis=1)
y = dataset['Sum']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

model = GradientBoostingRegressor(n_estimators=300)
model.fit(train_X, train_y)
joblib.dump(model, './balance_model.pkl')