from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

dataset = pd.read_csv("./velcom/training_set.csv")
dataset.head()

for column in dataset.columns:
  dataset[column].fillna(dataset[column].mean(), inplace=True)

X = dataset.drop(['churn'], axis=1)
y = dataset['churn']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

model = GradientBoostingClassifier(n_estimators=300)
model.fit(train_X, train_y)
joblib.dump(model, './model.pkl')
