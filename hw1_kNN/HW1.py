import matplotlib
matplotlib.use('Agg') 

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance


# returns array of [a, b] for k-fold kross-validation"
# a - array of points [x, y, c] for learning
# b - array of points [x, y, c] for testing
def getKrossValidatedData(k):
    a = []    
    fin = open("chips.txt", "r") 
    for line in fin.readlines():
        x, y, c = line.strip().split(",")
        x = float(x)
        y = float(y)
        c = int(c)
        a.append(dataTransform)
        #print(x, y, c)
    fin.close()
    random.shuffle(a)
    n = len(a) // k
    res = []
    for i in range(k):
        b = a[:n]
        a = a[n:]
        res.append([a[::], b[::]])
        a += b    
    return res


# returns array of [x, y, c, res]
# x, y 
# c - real category
# res - counted by k nearest neighbors category
def doKNN(dataSet, k, metric):
    k = min(k, len(dataSet[0]))
    res = [] 
    for p in dataSet[1]:
        a = [[metric(p[:2], x[:2]), x] for x in dataSet[0]]        
        a.sort()
        #print(a)
        a = [t[1] for t in a]                    
        s = 0
        for i in range(k):
            s += a[i][2]
        if (s > k // 2):
            s = 1
        else:
            s = 0
        res.append([p[0], p[1], p[2], s])
    return res


def safeResultsAsPng(kvNumber, neighborsNumber, data, metric):
    accErr = 0
    f1Err  = 0
    
    #dataSet = [points for learning, points for testing]
    #point = [x, y, c]   
    for i, dataSet in enumerate(data):     
        plt.subplot(2, 3, i + 1)
        plt.axis([-1, 1.5, -1, 1.5])

        #draw learning points with category 0
        a = [t[:2] for t in dataSet[0] if t[2] == 0]
        plt.plot(*zip(*a), marker='.', color='r', ls='')     
        #draw learning points with category 1    
        b = [t[:2] for t in dataSet[0] if t[2] == 1]
        plt.plot(*zip(*b), marker='.', color='b', ls='')
  
        testSet = doKNN(dataSet, neighborsNumber, metric)
        #    print(testSet)

        #draw guessed testing points with category 0
        c = [t[:2] for t in testSet if t[2] == t[3] == 0]
        plt.plot(*zip(*c), marker='o', color='r', ls='')
        #draw guessed testing points with category 1        
        d = [t[:2] for t in testSet if t[2] == t[3] == 1]    
        plt.plot(*zip(*d), marker='o', color='b', ls='')     
        
        #draw not guessed testing points with category 0        
        e = [t[:2] for t in testSet if t[2] != t[3] == 0]    
        plt.plot(*zip(*e), marker='s', color='r', ls='')
        #draw guessed testing points with category 1        
        f = [t[:2] for t in testSet if t[2] != t[3] == 1]    
        plt.plot(*zip(*f), marker='s', color='b', ls='')      
    
        plt.title(str(len(e) + len(f)) + " points of " + str(len(testSet)) + " was classified wrong") 
        accErr += (len(e) + len(f)) / len(testSet)
        
    accErr /= kvNumber
    
    s = "number of neigbors = " + str(neighborsNumber) + ", space = R^2, " + metric.__name__ + " distance\n"
    s += "average error for this 6-fold kross-validation is " + str(accErr) 
    plt.figure(1).suptitle(s, fontsize=14, fontweight='bold')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.savefig(metric.__name__ + "_" + str(neighborsNumber) +  '.png', dpi=100)  
    
    plt.close()

################################################################################
kvNumber = 6

data = getKrossValidatedData(kvNumber)

for neighborsNumber in range(1, 16):
    safeResultsAsPng(kvNumber, neighborsNumber, data, distance.euclidean)
    safeResultsAsPng(kvNumber, neighborsNumber, data, distance.cityblock)
#    safeResultsAsPng(kvNumber, neighborsNumber, data, distance.cosine)
#    safeResultsAsPng(kvNumber, neighborsNumber, data, distance.correlation)

