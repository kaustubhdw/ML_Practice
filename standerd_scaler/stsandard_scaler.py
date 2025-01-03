import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r"D:\ML_Practice\standerd_scaler\autompg.csv")

# Select features and target
X = df.iloc[:, [2, 4]]  # Selecting columns by indices
y = df.iloc[:, 5]       # Target column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)

# Fit the scaler on the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Scale training data

# Transform the test data using the same scaler
X_test = scaler.transform(X_test)       # Scale testing data

# Output mean of test data
print("Mean of X_test (scaled):", X_test.mean(axis=0))
print("Transformed X_test:")
print(X_test)

print("\nTransformed X_train:")
print(X_train)
