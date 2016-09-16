import numpy as np
import math
import random
import matplotlib.pyplot as plt


#returns array of [a, b] for k-fold kross-validation"
#
def getKrossValidation(k):
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
        res.append(a, b[::]])
        a += b    
    return res


def getDistance(a, b):
    d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return d

def doKNN(dataSet, k):
    
    return res
    
    
data = getKrossValidation(6)
for i, dataSet in enumerate(data):          
    plt.subplot(2, 3, i + 1)
    
    #points with category 0
    a = [t[:2] for t in dataSet[0] if t[2] == 0]
    plt.plot(*zip(*a), marker='o', color='r', ls='')     
    
    #points with category 1    
    b = [t[:2] for t in dataSet[0] if t[2] == 1]
    plt.plot(*zip(*b), marker='o', color='b', ls='')
  
    #points for testing
    testSet = doKNN(dataSet, 1)

    c = [t[:2] for t in testSet if t[2] == t[3] == 0]
    plt.plot(*zip(*c), marker='o', color='r', ls='')     
    
    d = [t[:2] for t in testSet if t[2] == t[3] == 1]    
    plt.plot(*zip(*d), marker='o', color='b', ls='')     
    
    e = [t[:2] for t in testSet if t[2] != t[3]]    
    plt.plot(*zip(*d), marker='*', color='y', ls='')      
    
    plt.title(len(e), "points classified wrong")    
    
    '''we need to calculate an average error for 6-fold kross-validation here'''
    

plt.show()