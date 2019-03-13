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

n_neighbors = 1
weights = 'uniform'
h = .02

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(W, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
#   plt.scatter(X[:, 0], X[:, 1], c=[0,0,0,0,1,1,1], cmap=cmap_bold) #,
plt.scatter([1,0,0,-1,0,0,-2], [0,1,-1,0,2,-2,0], c=[0,0,0,0,1,1,1], cmap=cmap_bold) #,
#               edgecolor='k', s=20)
plt.xlim(-2.5, 1.5)
plt.ylim(-2.5, 2.5)
plt.title("1-NN")
"""
"""
plt.show()



