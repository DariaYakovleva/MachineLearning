import numpy as np
import math
import random
import matplotlib.pyplot as plt


# returns array of [a, b] for k-fold Cross-validation"
# a - array of points [x, y, c] for learning
# b - array of points [x, y, c] for testing
def getCrossValidation(k):
    a = []    
    fin = open("chips.txt", "r") 
    for line in fin.readlines():
        x, y, c = line.strip().split(",")
        x = float(x)
        y = float(y)
        c = int(c)
        a.append([x, y, c])
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


def getDistance(a, b):
    d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return d


# returns array of [x, y, c, res]
# x, y - 
# c - real category
# res - counted by k nearest neighbors category
def doKNN(dataSet, k):
    k = min(k, len(dataSet[0]))
    res = [] 
    for p in dataSet[1]:
        a = [[getDistance(p[:2], x[:2]), x] for x in dataSet[0]]        
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

################################################################################
    
err = 0   

kvNumber = 6
neighborsNumber = 3

data = getCrossValidation(kvNumber)
for i, dataSet in enumerate(data):     
#    if i > 2:
#        plt.figure(2)
# creates new window with graphics

    plt.subplot(2, 3, i + 1)
    plt.axis([-1, 1.5, -1, 1.5])
    
    #dataSet = [points for learning, points for testing]
    #point = [x, y, c]
    
    #points with category 0
    a = [t[:2] for t in dataSet[0] if t[2] == 0]
    plt.plot(*zip(*a), marker='.', color='r', ls='')     
    
    #points with category 1    
    b = [t[:2] for t in dataSet[0] if t[2] == 1]
    plt.plot(*zip(*b), marker='.', color='b', ls='')
  
    testSet = doKNN(dataSet, neighborsNumber)
    print(testSet)

    c = [t[:2] for t in testSet if t[2] == t[3] == 0]
    plt.plot(*zip(*c), marker='o', color='r', ls='')     
    
    d = [t[:2] for t in testSet if t[2] == t[3] == 1]    
    plt.plot(*zip(*d), marker='o', color='b', ls='')     
    
    e = [t[:2] for t in testSet if t[2] != t[3] == 0]    
    plt.plot(*zip(*e), marker='s', color='r', ls='')
    
    f = [t[:2] for t in testSet if t[2] != t[3] == 1]    
    plt.plot(*zip(*f), marker='s', color='b', ls='')      
    
    plt.title(str(len(e) + len(f)) + " points of " + str(len(testSet)) + " was classified wrong") 
    
    err += (len(e) + len(f)) / len(testSet)
    
s = "number of neigbors = " + str(neighborsNumber) + ", space = R^2, Euclidean distance\n"
s += "average error for this 6-fold Cross-validation is " + str(err / kvNumber)
plt.figure(1).suptitle(s, fontsize=14, fontweight='bold')

plt.show()

#TODO
result = pd.DataFrame(columns=['folds', 'kNN', 'metric', 'accuracy', 'transformation'])
transformations = [lambda x: x ** 2, lambda x: x ** 0.5]
metrics = [euclidean, cityblock, cosine, correlation]
for fold in range(3, 10):
	for neighbors in range(2, 10):
	    for metric in metrics:
    	    for transform in transformations:
    	    	data = getCrossValidation(fold)
    	    	curData = data[['x', 'y']].applymap(transform).join(data['class'])
				testSet = doKNN(data, neighbors)       	 
       	     	k, accuracy = training(curData, fold, metric)
            	result = result.append([fold, k, str(metric), accuracy, str(transform)], ignore_index=True)
print(result)  