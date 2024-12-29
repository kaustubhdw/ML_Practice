import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("autompg.csv")
print(df.head())
print(df.dtypes)

X = df.iloc[:,[2,4]]
y = df.iloc[:,[5]]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)
scaler = MinMaxScaler().fit(X_train)
print(scaler.data_min_)
print(scaler.data_max_)

X_train.describe()
print(scaler.feature_range)
X_train = scaler.transform(X_train)
scaler = MinMaxScaler().fit(X_test)
X_test = scaler.transform(X_test)