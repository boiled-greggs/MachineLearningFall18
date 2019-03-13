"""
logisic regression for classifying handwritten ones and fives.
ones are +1 and fives are -1.
"""
import numpy as np
import matplotlib.pyplot as plt

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
        X.append([1, I_ones[i], S_ones[i]])
        Y.append(1)
    for i in range(len(I_fives)):
        X.append([1, I_fives[i], S_fives[i]])
        Y.append(-1)

    return X, Y

def sum_gradient(X, Y, w):
    gt = [0, 0, 0]
    for i in range(len(X)):
        factor = Y[i] / (1 + np.exp(Y[i]*np.dot(w, X[i])))
        gt[0] += factor * X[i][0]
        gt[1] += factor * X[i][1]
        gt[2] += factor * X[i][2]

    gt = [-1./len(X)*gj for gj in gt]
    return gt

def descent(X, Y, w, t, eta, condition):
    while(t < 1000):
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

if __name__=="__main__":
    I_ones, S_ones, I_fives, S_fives = get_ones_fives("./zip.train")
    X, Y = get_D(I_ones, S_ones, I_fives, S_fives)
    w = np.array([0, 0, 0])
    t = 0
    eta = 2.
    condition = .1

    w_f = descent(X, Y, w, t, eta, condition)
    print(w_f)

#   linx_arr = np.linspace(-.9, -.175, 1000)
#   y_arr = [-(w_f[0] + x*w_f[1])/w_f[2] for x in linx_arr]
    
    E_in = get_E_in(X, Y, w_f)
    print("E_in =", E_in)

#   fig, ax = plt.subplots()
#   on = ax.scatter(I_ones, S_ones,s=60, facecolors='none', edgecolors='b')
#   fi = ax.scatter(I_fives, S_fives, marker='x', c='r')
#   plt.xlabel("Average Intensity of Digit")
#   plt.ylabel("Average Reflection Symmetry of Digit")
#   ax.plot(linx_arr, y_arr)
#   plt.legend((on, fi),("Ones", "Fives"))
#   fig.savefig('trainingplot.png', bbox_inches='tight')

    I_ones_t, S_ones_t, I_fives_t, S_fives_t = get_ones_fives("./ZipDigits.test")
    Xt, Yt = get_D(I_ones_t, S_ones_t, I_fives_t, S_fives_t)
    E_test = get_E_in(Xt, Yt, w_f)
    print("E_test =", E_test)

#   fig, ax = plt.subplots()
#   on = ax.scatter(I_ones_t, S_ones_t,s=60, facecolors='none', edgecolors='b')
#   fi = ax.scatter(I_fives_t, S_fives_t, marker='x', c='r')
#   plt.xlabel("Average Intensity of Digit")
#   plt.ylabel("Average Reflection Symmetry of Digit")
#   ax.plot(linx_arr, y_arr)
#   plt.legend((on, fi),("Ones", "Fives"))
#   fig.savefig("testplot.png", bbox_inches='tight')
