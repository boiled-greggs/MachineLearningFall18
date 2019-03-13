"""
logisic regression for classifying handwritten ones and fives.
ones are +1 and fives are -1.
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import var, plot_implicit

def get_intensity(digit):
    return np.average(digit)

def get_symmetry(digit):
    thing = []
    for i in range(16):
        for j in range(8):
            thing.append(digit[i*16+j] - digit[i*16 + 15 - j])

    return np.average(thing)

def get_ones_fives(filename):
    f = open(filename)
    ones = []
    fives = []
    for line in f:
        digit = line.split()
        if (int(float(digit[0])) == 1):
            del digit[0]
            ones.append(line.split())
        elif (int(float(digit[0])) == 5):
            del digit[0]
            fives.append(line.split())
    
    ones = np.array(ones) #ones.astype(np.float)
    fives = np.array(fives) #fives.astype(np.float)
    fones = ones.astype(np.float)
    ffives = fives.astype(np.float)

    I_ones = []
    S_ones = []
    I_fives = []
    S_fives = []

    for one in fones:
        I_ones.append(get_intensity(one))
        S_ones.append(get_symmetry(one))
    for five in ffives:
        I_fives.append(get_intensity(five))
        S_fives.append(get_symmetry(five))

    return I_ones, S_ones, I_fives, S_fives

def get_D(I_ones, S_ones, I_fives, S_fives):
    X = []
    Y = []
    for i in range(len(I_ones)):
        X.append([1, I_ones[i], S_ones[i],
            I_ones[i]**2, S_ones[i]**2, I_ones[i]*S_ones[i],
            I_ones[i]**3, S_ones[i]**3, S_ones[i]*I_ones[i]**2, I_ones[i]*S_ones[i]**2])
        Y.append(1)
    for i in range(len(I_fives)):
        X.append([1, I_fives[i], S_fives[i],
            I_fives[i]**2, S_fives[i]**2, I_fives[i]*S_fives[i],
            I_fives[i]**3, S_fives[i]**3, S_fives[i]*I_fives[i]**2, I_fives[i]*S_fives[i]**2])
        Y.append(-1)

    return X, Y

def sum_gradient(X, Y, w):
    gt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(X)):
        factor = Y[i] / (1 + np.exp(Y[i]*np.dot(w, X[i])))
        gt[0] += factor * X[i][0]
        gt[1] += factor * X[i][1]
        gt[2] += factor * X[i][2]
        gt[3] += factor * X[i][3]
        gt[4] += factor * X[i][4]
        gt[5] += factor * X[i][5]
        gt[6] += factor * X[i][6]
        gt[7] += factor * X[i][7]
        gt[8] += factor * X[i][8]
        gt[9] += factor * X[i][9]
        

    gt = [-1./len(X) * gj for gj in gt]
    return gt

def descent(X, Y, w, t, eta, condition):
    while(t < 5000):
        gt = sum_gradient(X, Y, w)
        vt = [-1.*gti for gti in gt]
        wt1 = w + [eta*vi for vi in vt]
        w = wt1
        t += 1
    return w

def get_E_in(X, Y, w):
    E_in = 0
    for i in range(len(X)):
        classif = Y[i]
        y_classified = -(w[0] + w[1]*X[i][1])/w[2]
        y_actual = X[i][2]
        if (classif == 1):
            if (y_actual >= y_classified):
                E_in += 1
        elif (classif == -1):
            if (y_actual <= y_classified):
                E_in += 1
    return E_in/float(len(Y))

def get_E_in_third(X, Y, w):
    E_in = 0
    for i in range(len(X)):
        x1 = X[i][1]
        x2 = X[i][2]
        z = w[0] + w[1]*x1 + w[2]*x2 + w[3]*x1**2 + w[4]*x2**2 + w[5]*x1*x2 + \
                w[6]*x1**3 + w[7]*x2**3 + w[8]*x2*x1**2 + w[9]*x1*x2**2
        if (Y[i] == 1 and z <= 0):
            E_in += 1
        if (Y[i] == -1 and z >= 0):
            E_in += 1

    return E_in/float(len(Y))

def plot(I_ones, S_ones, I_fives, S_fives, w):
    xarange = np.linspace(-1.0, 0.0, 1000)
    yarange = np.linspace(-1.0, 0.2, 1000)
    X, Y = np.meshgrid(xarange,yarange)
    Z = w[1]*X + w[2]*Y + w[3]*X**2 + w[5]*X*Y + w[4]*Y**2 + \
            w[6]*X**3 + w[7]*Y**3 + w[9]*X*Y**2 + w[8]*Y*X**2
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            Z[i][j] += w[0]
    print(Z)
    fig, ax = plt.subplots()
    on = ax.scatter(I_ones, S_ones,s=60, facecolors='none', edgecolors='b')
    fi = ax.scatter(I_fives, S_fives, marker='x', c='r')
    ax.contour(X, Y, Z, [0.0])
    ax.set_xlim([-.95,0])
    ax.set_ylim([-.95, .2])
    plt.xlabel("Average Intensity of Digit")
    plt.ylabel("Average Reflection Symmetry of Digit")
    plt.legend((on, fi),("Ones", "Fives"))
#   fig.savefig("3rdtransform.png", bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    I_ones, S_ones, I_fives, S_fives = get_ones_fives("./zip.train")
    X, Y = get_D(I_ones, S_ones, I_fives, S_fives)
    w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    t = 0
    eta = 5
    condition = .1

#   w_f = descent(X, Y, w, t, eta, condition)
#   print(w_f)

    w_f = [-10.74014039, -5.031538,   -8.65046953,  5.48060507, -5.33501925, \
        14.16533797, -3.05653689, 10.35446363,-11.90702955, -3.19098579]
    w = w_f

    E_in = get_E_in_third(X, Y, w)
    print("E_in =", E_in)

    I_ones_t, S_ones_t, I_fives_t, S_fives_t = get_ones_fives("./ZipDigits.test")
    Xt, Yt = get_D(I_ones_t, S_ones_t, I_fives_t, S_fives_t)
    E_test = get_E_in_third(Xt, Yt, w)
    print("E_test =", E_test)

    plot(I_ones_t, S_ones_t, I_fives_t, S_fives_t, w)
