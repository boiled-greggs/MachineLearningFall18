import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

def new_D(N, rad_i, rad_o, sep, red, blue):
    random.seed()
    D = []
    y = []
    shiftx = 8 # -(rad_i+thk/2)/2
    shifty = 8 # sep/2
    for i in range(N):
        r = random.uniform(rad_i, rad_o)
        theta = random.uniform(0, 2*np.pi)
        x1 = r*np.cos(theta) + shiftx
        x2 = r*np.sin(theta) + shifty
        yn = 0
        if (theta >= np.pi and theta < 2*np.pi):
            x1 += (rad_i + rad_o)/2
            x2 -= sep
            yn = blue
        else: 
            yn = red
        D.append([x1, x2])
        y.append(yn)

    return D, y

if __name__ == "__main__":
    inner_rad = 10
    thk = 5
    outer_rad = inner_rad + thk
    sep = 5
    red = -1
    blue = 1
    N = 2000
    X, y = new_D(N, inner_rad, outer_rad, sep, red, blue)
    X = np.matrix(X)
    y = np.array(y)
    print(X)
    X1 = [xi[0] for xi in X.tolist()]
    X2 = [xi[1] for xi in X.tolist()]
    print(X1)

    n_neighbors = 3
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
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
#   plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold) #,
    plt.scatter(X1, X2, c=y, cmap=cmap_bold, s=10) #,
#   plt.scatter([1,0,0,-1,0,0,-2], [0,1,-1,0,2,-2,0], c=[0,0,0,0,1,1,1], cmap=cmap_bold) #,
#               edgecolor='k', s=20)
    plt.title("3-NN")

    plt.show()
