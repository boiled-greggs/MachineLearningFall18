import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

    
X = [[1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0]]
y = [-1, -1, -1, -1, 1, 1, 1]

W = []
for x in X:
    z1 = np.sqrt(x[0]**2 + x[1]**2)
    z2 = 0
    if (x[0] != 0):
        z2 = np.arctan(x[1] / x[0])
    else: z2 = np.pi/2
    W.append([z1,z2])

W = np.matrix(W)
X = np.matrix(X)
h = .02

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
print(x_min, x_max)
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.zeros(xx.shape)
rows, cols = Z.shape
for i in range(rows):
    for j in range(cols):
        minind = 0
        minval = 100
        for k in range(X.shape[0]):
            dist = np.sqrt((xx[i,j] - X[k,0])**2 + (yy[i,j] - X[k,1])**2)
            if (dist < minval): 
                minval = dist
                minind = k
        Z[i,j] = y[minind]


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.figure()
#Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.show()
