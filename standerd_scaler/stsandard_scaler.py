import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("autompg.csv")

X = df.iloc[:,[2,4]]
y = df.iloc[:,5]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 40)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

scaler = StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)
X_test.mean(axis = 0)

print(X_test)
print(X_train)