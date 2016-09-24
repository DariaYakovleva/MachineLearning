
# coding: utf-8

# In[45]:

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
from IPython.display import display, HTML


# In[2]:

def getData():
    data = pd.read_table("D:/Users/Daria/Code2/MachineLearning/hw1_kNN/chips.txt", sep=",", header=None, names=['x', 'y', 'class'])
    return data


# In[39]:

# k Nearest Neighbors algorithm used for classification
def kNN(k, metrics, train, test):
    k = min(k, len(train))
    result = []
    for point in test:
        distances = [[int(tpoint[2]), metrics(point[:2], tpoint[:2])] for tpoint in train]
        distances.sort(key=lambda p:p[1])
        #nlargest
        count0 = 0
        count1 = 1
        for i in range(k):
            if (distances[0] == 0):
                count0 = count0 + 1
            else:
                count1 = count1 + 1
        prediction = 1
        if (count0 > count1):
            prediction = 0
        result.append([point[0], point[1], int(point[2]), prediction])
    return result


# In[42]:

# k-fold cross-validation
def training(data, k, metrics, kNN_value):
    kf = KFold(len(data), n_folds=k, shuffle=True)
    averageAccuracy = 0
    for train_index, test_index in kf:
        train = data.loc[train_index].as_matrix()
        test = data.loc[test_index].as_matrix()
        result = kNN(kNN_value, metrics, train, test)
        #f1 measure =  2 * (precision * recall) / (precision + recall)
        f1 = f1_score([row[2] for row in test], [row[3] for row in result]) 
        averageAccuracy += f1
    averageAccuracy /= k
    return averageAccuracy


# In[53]:

data = getData()
# cross_val_score(log_reg, X_train, y_train, cv=5)
# sklearn.metrics.classification_report(y_test, log_reg.predict(X_test))
result = pd.DataFrame(columns=['kNN', 'folds', 'metric', 'transformation', 'accuracy'])
transformations = [lambda x: x]#, lambda x: x ** 2, lambda x: x ** 0.5]
metrics = [cityblock, euclidean, cityblock, cosine, correlation]

#find the best value of k
for k in range(1, 20, 2): #len(train) // 2):
    for fold in [10]:
        for metric in metrics:
            for transform in transformations:
                curData = data[['x', 'y']].applymap(transform).join(data['class'])
                accuracy = training(curData, fold, metric, k)
                cur = pd.DataFrame([[k, fold, str(metric).split(' ')[1], 'x->x', accuracy]], columns=['kNN', 'folds', 'metric', 'transformation', 'accuracy'])
                result = result.append(cur, ignore_index=True)
display(result)
showPoints(result['kNN'], result['accuracy'], 'b', '-')


# In[48]:

def showPoints(x_ticks, y_ticks, color, style):
    #plt.subplot(k, 1, num)
    #plt.figure(1).suptitle('neighbors = ' + str(optimalKNN) + '; folds = ' + str(k) + '; metrics = ' + str(metrics), fontsize=14, fontweight='bold')
    #plt.title('accuracy = ' + str(f1))
    plt.plot(x_ticks, y_ticks, color + style)
    plt.show()


# In[ ]:

data0 = train[train['class'] == 0]
    data0.append(test[(test['class'] == test['prediction']) & (test['class'] == 0)])
    data1 = train[train['class'] == 1]
    data1.append(test[(test['class'] == test['prediction']) & (test['class'] == 1)])
    data2 = test[test['class'] != test['prediction']]

