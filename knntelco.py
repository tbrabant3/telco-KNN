import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


dataset = pd.read_csv("telco_data.csv", delimiter=',')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Transform the TotalCharges column to a column with numbers instead of a space
dataset.replace(r"^\s*$", 0, regex=True, inplace=True)
dataset["TotalCharges"] = dataset["TotalCharges"].astype('float64')

# Drop specific columns that don't matter
X = X.drop(columns=['customerID'])

# Encode the data
label_encoder = preprocessing.LabelEncoder()
for col in X.columns:
    X[col] = label_encoder.fit_transform(X[col])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Scale / Normalize data
scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


amount_of_neighbors = range(1, 10, 1)
train_acc = []
test_acc = []
for k in amount_of_neighbors:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    train_acc.append(classifier.score(X_train, y_train))
    test_acc.append(classifier.score(X_test, y_test))


plt.title('Churn Predictions')
plt.plot(amount_of_neighbors, test_acc, label='Testing Accuracy')
plt.plot(amount_of_neighbors, train_acc, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()




