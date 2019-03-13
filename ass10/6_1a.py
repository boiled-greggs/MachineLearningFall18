import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 1

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = np.matrix('1 0; 0 1; 0 -1; -1 0; 0 2; 0 -2; -2 0') # iris.data[:, :2]
# y = [-1, -1, -1, -1, 1, 1, 1] # iris.target
y = np.array([ 0., 0., 0., 0., 1., 1., 1.])

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

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
