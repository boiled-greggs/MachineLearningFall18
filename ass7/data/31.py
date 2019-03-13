import numpy as np
import matplotlib.pyplot as plt
import random

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
        D.append([1, x1, x2])
        y.append(yn)

    return D, y

def is_done(X, y, w):
    for i in range(len(X)):
        if (np.sign(np.dot(w, X[i])) != y[i]):
            return i, False
    return -1, True

def PLA(X, y, w, inner_rad, outer_rad, thk, sep, N):
    it = 0
    while(True):
        it += 1
        loc, done = is_done(X, y, w)
        if (done):
            break
        else:
            w = np.add(w, [y[loc]*x for x in X[loc]])
    
    return it

def plot(X, w, inner_rad, outer_rad):
    x1s = [x[1] for x in X]
    x2s = [x[2] for x in X]

    x_array = np.linspace(-(outer_rad+inner_rad), outer_rad+inner_rad, 1000)
    y_array = [(w[0] + x*w[1])/w[2] for x in x_array]

    fig, ax = plt.subplots()
    ax.scatter(x1s,x2s,marker='.')
    ax.plot(x_array, y_array)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.show()

def linear_reg(X,y, outer_rad, inner_rad):
    X_m = np.matrix([x for x in X])
    X_cross = (X_m.getT()*X_m).getI()*X_m.getT()
    y = np.matrix([y[i] for i in range(len(y))])
    w_lin = X_cross*y.getT()

    w = [w_lin[0].item(), w_lin[1].item(), w_lin[2].item()]

    x1s = [x[1] for x in X]
    x2s = [x[2] for x in X]

    x_array = np.linspace(-(outer_rad+inner_rad), outer_rad+inner_rad, 1000)
    y_array = [(w[0] + x*w[1])/w[2] for x in x_array]

    fig, ax = plt.subplots()
    ax.scatter(x1s,x2s,marker='.')
    ax.plot(x_array, y_array)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.show()


if __name__=="__main__":
    inner_rad = 10
    thk = 5
    outer_rad = inner_rad + thk
    sep = 5
    red = -1
    blue = 1
    N = 2000
    seps = np.arange(0.2, 5.1, 0.1)

    num_its = []
    for sepa in seps:
        X, y = new_D(N, inner_rad, outer_rad, sep, red, blue)
        w = [0,0,0]

        num_its.append(PLA(X, y, w, inner_rad, outer_rad, thk, sep, N))

    fix, ax = plt.subplots()
    ax.plot(seps, num_its)
    plt.show()
#   linear_reg(X, y, outer_rad, inner_rad)


