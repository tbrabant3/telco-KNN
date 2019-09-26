
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

import numpy as np
import random
import math
import operator


def loadDataset(file, split, training_set, test_set):
    iris_data = np.genfromtxt(file, dtype=None, encoding="UTF-8", delimiter=",")
    for x in range(len(iris_data) - 1):
        for y in range(4):
            iris_data[x][y] = float(iris_data[x][y])
        if random.random() < split:
            training_set.append(iris_data[x])
        else:
            test_set.append(iris_data[x])


def euclideanDistance(instance_1, instance_2, length):
    distance = 0
    for x in range(length):
        distance += (instance_1[x] - instance_2[x]) ** 2
    return math.sqrt(distance)


def getNeighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        euc_dist = euclideanDistance(test_instance, training_set[x], length)
        distances.append((training_set[x], euc_dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for y in range(k):
        neighbors.append(distances[y][0])
    return neighbors


def getResponse(neighbors):
    items = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in items:
            items[response] += 1
        else:
            items[response] = 1
    sortedItems = sorted(items.items(), key=operator.itemgetter(1), reverse=True)
    return sortedItems[0][0]


def getAccuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100


def main():
    training_set = []
    test_set = []
    split = 0.67
    loadDataset("iris.txt", split, training_set, test_set)
    print("Training set length: ", len(training_set))
    print("Test set length: ", len(test_set))

    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = getNeighbors(training_set, test_set[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print("Predicted = " + result + ", Actual = " + test_set[x][-1])
    accuracy = getAccuracy(test_set, predictions)
    print("Accuracy: " + repr(accuracy) + "%")


main()

