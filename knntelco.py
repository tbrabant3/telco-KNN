# Author: Tyler Brabant
# Class:  CSI-270-01
# Certification of Authenticity:
# I certify that this is entirely my own work, except where I have given fully documented
# references to the work of others.  I understand the definition and consequences of
# plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
# assessing this assignment reproduce this assignment and provide a copy to anothermember
# of academic staff and / or communicate a copy of this assignment to a plagiarism checking
# service(which may then retain a copy of this assignment on its database for the purpose
# of future plagiarism checking).
# This code has been based off of a tutorial on MachineLearningMastery
# Some of the components have been modified

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


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


# amount_of_neighbors = range(1, 10, 1)
test_acc = []

# Classifier to setup amount of neighbors and weighting type
classifier = KNeighborsClassifier(n_neighbors=2, weights="uniform")
classifier.fit(X_train, y_train)

test_acc.append(classifier.score(X_test, y_test))

# Use the training data to train the data then predict
print(test_acc)




