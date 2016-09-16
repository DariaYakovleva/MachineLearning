import numpy as np
import math
import matplotlib.pyplot as plt


fin = open("chips.txt", "r")

a0 = []
a1 = []
for line in fin.readlines():
    (x, y, c) = line.strip().split(",")
    x = float(x)
    y = float(y)
    #x, y =  x**2 + y**2, x/y
    #print(x, y, c)
    if (c == '0'):
        a0.append([x, y])
    else:
        a1.append([x, y])
    
    
plt.plot(*zip(*a0), marker='o', color='r', ls='')
plt.plot(*zip(*a1), marker='o', color='b', ls='')
plt.show()