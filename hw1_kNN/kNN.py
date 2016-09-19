import numpy as np
import pandas as pd
import math
import random
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import euclidean, cityblock, cosine, correlation

def getData():
    data = pd.read_table("D:/Users/Daria/Code2/MachineLearning/hw1_kNN/chips.txt", sep=",", header=None, names=['x', 'y', 'class'])
    return data

def showPoints(train, test):
    data0 = train[train['class'] == 0]
    data0.append(test[(test['class'] == test['prediction']) & (test['class'] == 0)])
    data1 = train[train['class'] == 1]
    data1.append(test[(test['class'] == test['prediction']) & (test['class'] == 1)])
    data2 = test[test['class'] != test['prediction']]
    plt.plot(data0['x'], data0['y'], "go")
    plt.plot(data1['x'], data1['y'], "bo")
    plt.plot(data2['x'], data2['y'], "ro")
    return len(data2)

def kNN(k, metrics, train, test):
    k = min(k, len(train))
    result = pd.DataFrame(columns=['x', 'y', 'class', 'prediction'])
    for index, point in test.iterrows():
        distances = [[int(tpoint['class']), metrics(point[['x', 'y']], tpoint[['x', 'y']])] for index, tpoint in train.iterrows()]
        distances.sort(key=lambda p:p[1])
        #nlargest
        count0 = 0
        count1 = 1
        for i in range(k):
            if (distances[0] == 0):
                count0 = count0 + 1
            else:
                count1 = count1 + 1
        if (count0 > count1):
            point['prediction'] = 0
        else:
            point['prediction'] = 1
        result = result.append(point, ignore_index=True)
    return result

def training(data, k, metrics):
    kf = KFold(len(data), n_folds=k, shuffle=True)
    errors = len(data)
    num = 0
    averageK = 0
    averageAccuracy = 0
    for train_index, test_index in kf:
        train = data.loc[train_index]
        test = data.loc[test_index]
        accuracy = 0
        optimalKNN = 0
        #for kNN_value in range(1, len(train) // 2):
        for kNN_value in range(1, 10):
            print(kNN_value)
            result = kNN(kNN_value, metrics, train, test)
            f1 = f1_score(test.as_matrix(['class']), result.as_matrix(['prediction']))
            if (f1 > accuracy):
                accuracy = f1
                optimalKNN = kNN_value
        result = kNN(optimalKNN, metrics, train, test)
        f1 = f1_score(test.as_matrix(['class']), result.as_matrix(['prediction']))
        averageK += optimalKNN
        averageK += f1
        #num = num + 1
        #plt.subplot(k, 1, num)
        #plt.figure(1).suptitle('neighbors = ' + str(optimalKNN) + '; folds = ' + str(k) + '; metrics = ' + str(metrics), fontsize=14, fontweight='bold')
        #error = showPoints(train, result)
        #plt.title('accuracy = ' + str(f1))
        #errors = min(errors, error)
    #plt.show()
    averageK /= k
    averageAccuracy /= k
    return averageK, averageAccuracy    

data = getData()
result = pd.DataFrame(columns=['folds', 'kNN', 'metric', 'accuracy']) #result table
metrics = [euclidean, cityblock, cosine, correlation]
for fold in range(3, 11):
    for metric in metrics:
        k, accuracy = training(data, fold, metric)
        result = result.append([fold, k, str(metric), accuracy], ignore_index=True)
print(result)    