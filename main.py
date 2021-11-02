import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

def encode_winner(datapoint):
    if datapoint == 'Blue':
        return 0
    else:
        return 1


df = pd.read_csv('data/preprocessed_data.csv')
pd.set_option('display.max_columns', None)
df['Winner'] = df['Winner'].apply(lambda x: encode_winner(x))

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

pipeline = make_pipeline(PolynomialFeatures(1), Lasso(alpha=1e-12))
pipeline.fit(X_train, y_train)
train_err = mean_squared_error(y_train, pipeline.predict(X_train))
test_err = mean_squared_error(y_test, pipeline.predict(X_test))
print(pipeline.predict(X_train), y_train)
print(f"MSE TRAIN: {train_err}")
print(f"MSE TEST: {test_err}")

# 0 - BLUE | 1 - RED

