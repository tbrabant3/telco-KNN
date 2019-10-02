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

def loadDataset(file):
    dataset = pd.read_csv(file, delimiter=',')
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Transform the TotalCharges column to a column with numbers instead of a space
    dataset.replace(r"^\s*$", 0, regex=True, inplace=True)
    dataset["TotalCharges"] = dataset["TotalCharges"].astype('float64')

    # Drop specific columns that don't matter
    X = X.drop(columns=['customerID'])
    return X, y


def encode(X):
    # Encode the data
    label_encoder = preprocessing.LabelEncoder()
    for col in X.columns:
        X[col] = label_encoder.fit_transform(X[col])


def standardScale(test, train):
    # Scale / Normalize data
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)
    return train, test

def main():
    X, y = loadDataset("telco_data.csv")
    encode(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
    X_train, X_test = standardScale(X_test, X_train)

    # Classifier to setup amount of neighbors and weighting type
    classifier = KNeighborsClassifier(n_neighbors=3, weights="distance")
    classifier.fit(X_train, y_train)

    print(classifier.score(X_test, y_test))


if __name__ == '__main__':
    main()
