'''
# Author: Ashton Allen
-- Most of this work came from this tutorial: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# Class:  CSI-480-01
# Certification of Authenticity:
# I certify that this is entirely my own work, except where I have given fully documented
# references to the work of others.  I understand the definition and consequences of
# plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
# assessing this assignment reproduce this assignment and provide a copy to anothermember
# of academic staff and / or communicate a copy of this assignment to a plagiarism checking
# service(which may then retain a copy of this assignment on its database for the purpose
# of future plagiarism checking).
'''

import operator
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def load_dataset(filename, split, training_set=[], test_set=[]):
    dataset = pd.read_csv(filename)
    dataframe = pd.DataFrame(dataset).drop(columns=['customerID'])

    for col in dataframe.columns:
        if dataframe[col].dtype != np.number:
            le = LabelEncoder()
            le.fit(dataframe[col])
            dataframe[col] = le.transform(dataframe[col])

    dataframe = shuffle(dataframe)
    count_row = dataframe.shape[0]
    split_point = int(np.ceil(count_row * split))

    train = dataframe.iloc[:split_point]
    test = dataframe.iloc[split_point:]

    # Iterate over each row and append
    for index, rows in train.iterrows():
        training_set.append(rows)

    for index, rows in test.iterrows():
        test_set.append(rows)


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return np.sqrt(distance)


def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    training_set = []
    test_set = []
    split = 0.67
    load_dataset('Telco-Customer-Churn_training_data.csv', split, training_set, test_set)
    print('Train set: ' + repr(len(training_set)))
    print('Test set: ' + repr(len(test_set)))


    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('predicted=' + repr(result) + ', actual=' + repr(test_set[x][-1]))
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()
