import numpy as np
import matplotlib.pyplot as plt
import random


def get_intensity(digit):
    return (np.average(digit) + .32022178989)/0.5521245136254086

def get_symmetry(digit):
    thing = []
    for i in range(16):
        for j in range(8):
            thing.append(digit[i*16+j] - digit[i*16 + 15 - j])

    return (np.average(thing) + .30121484375)/0.58981640625

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
    total_size = len(I_ones) + len(I_fives)
    indices_to_sample = random.sample(range(0, total_size), 300)
    I_ones_sampled = []
    S_ones_sampled = []
    I_fives_sampled = []
    S_fives_sampled = []
    for i in indices_to_sample:
        if (i < len(I_ones)):
            X.append(transform(I_ones[i], S_ones[i]))
            Y.append(1)
            I_ones_sampled.append(I_ones[i])
            S_ones_sampled.append(S_ones[i])
        else:
            X.append(transform(I_fives[i-len(I_ones)], S_fives[i-len(I_ones)]))
            Y.append(-1)
            I_fives_sampled.append(I_fives[i-len(I_ones)])
            S_fives_sampled.append(S_fives[i-len(I_ones)])

    return X, Y, I_ones_sampled, S_ones_sampled, I_fives_sampled, S_fives_sampled

def legendre(L_km2, L_km1, k, k_f, x):
    while(k <= k_f):
        L_k = ((2.*k - 1.)/k)*3.*L_km1 - ((k-1.)/k)*L_km2
        L_km2 = L_km1
        L_km1 = L_k
        k += 1
    return L_k

def transform(x1, x2):
    L0 = 1.
    L1x1 = x1
    L1x2 = x2
    L2x1 = legendre(L0, L1x1, 2., 2., x1)
    L2x2 = legendre(L0, L1x2, 2., 2., x2)
    L3x1 = legendre(L1x1, L2x1, 3., 3., x1)
    L3x2 = legendre(L1x2, L2x2, 3., 3., x2)
    L4x1 = legendre(L2x1, L3x1, 4., 4., x1)
    L4x2 = legendre(L2x2, L3x2, 4., 4., x2)
    L5x1 = legendre(L3x1, L4x1, 5., 5., x1)
    L5x2 = legendre(L3x2, L4x2, 5., 5., x2)
    L6x1 = legendre(L4x1, L5x1, 6., 6., x1)
    L6x2 = legendre(L4x2, L5x2, 6., 6., x2)
    L7x1 = legendre(L5x1, L6x1, 7., 7., x1)
    L7x2 = legendre(L5x2, L6x2, 7., 7., x2)
    L8x1 = legendre(L6x1, L7x1, 8., 8., x1)
    L8x2 = legendre(L6x2, L7x2, 8., 8., x2)
    return [L0, L1x1, L1x2, L2x1, L1x1*L1x2, L2x2, L3x1, L2x1*L1x2, L1x1*L2x2, L3x2, \
            L4x1, L3x1*L1x2, L2x1*L2x2, L1x1*L3x2, L4x2, L5x1, L4x1*L1x2, L3x1*L2x2, L2x1*L3x2, \
            L1x1*L4x2, L5x2, L6x1, L5x1*L1x2, L4x1*L2x2, L3x1*L3x2, L2x1*L4x2, L1x1*L5x2, \
            L6x2, L7x1, L6x1*L1x2, L5x1*L2x2, L4x1*L3x2, L3x1*L4x2, L2x1*L5x2, L1x1*L6x2, \
            L7x2, L8x1, L7x1*L1x2, L6x1*L2x2, L5x1*L3x2, L4x1*L4x2, L3x1*L5x2, L2x1*L6x2, \
            L1x1*L7x2, L8x2]

def get_w(Z, Z_T, I, y, lamb):
    matrix = Z_T*Z + lamb*I
    matrixI = matrix.getI()
    second = matrixI*Z_T
    wreg = second*y
    # wreg = ((Z_T*Z + lamb*I).getI() * Z_T) * y
    return wreg

def plot(wreg, I_ones, S_ones, I_fives, S_fives):
    xarange = np.linspace(-1.0, 1.0, 2000)
    yarange = np.linspace(-1.0, 1.0, 2000)
    X, Y = np.meshgrid(xarange,yarange)
    ones = np.ones((2000,2000))
    w = []
    for wi in wreg.tolist():
        w.append(wi[0])
    print(w)
    L0 = ones
    L1x1 = X
    L2x1 = (2*2-1)/2.*X*L1x1 - (2-1)/2.*L0 #1.5*X**2 - .5*ones
    L3x1 = (2*3-1)/3.*X*L2x1 - (3-1)/3.*L1x1 #2.5*X**3 - 1.5*X
    L4x1 = (2*4-1)/4.*X*L3x1 - (4-1)/4.*L2x1 #4.375*X**4 - 3.75*X**2 + .125*3*ones
    L5x1 = (2*5-1)/5.*X*L4x1 - (5-1)/5.*L3x1 #7.875*X**5 - 8.75*X**3 + 1.875*X
    L6x1 = (2*6-1)/6.*X*L5x1 - (6-1)/6.*L4x1 #14.4375*X**6 - 19.6875*X**4 + 6.5625*X**2 - 5/16*ones
    L7x1 = (2*7-1)/7.*X*L6x1 - (7-1)/7.*L5x1 #26.8125*X**7 - 43.3125*X**5 + 19.6875*X**3 - 2.1875*X
    L8x1 = (2*8-1)/8.*X*L7x1 - (8-1)/8.*L6x1 #50.2734375*X**8 - 93.84375*X**6 + 54.140625*X**4 - 9.84375*X**2 + 35/128*ones
    L1x2 = Y
    L2x2 = (2*2-1)/2.*Y*L1x2 - (2-1)/2.*L0 #1.5*Y**2 - .5*ones
    L3x2 = (2*3-1)/3.*Y*L2x2 - (3-1)/3.*L1x2 #2.5*Y**3 - 1.5*Y
    L4x2 = (2*4-1)/4.*Y*L3x2 - (4-1)/4.*L2x2 #4.375*Y**4 - 3.75*Y**2 + .125*3*ones
    L5x2 = (2*5-1)/5.*Y*L4x2 - (5-1)/5.*L3x2 #7.875*Y**5 - 8.75*Y**3 + 1.875*Y
    L6x2 = (2*6-1)/6.*Y*L5x2 - (6-1)/6.*L4x2 #14.4375*Y**6 - 19.6875*Y**4 + 6.5625*Y**2 - 5/16*ones
    L7x2 = (2*7-1)/7.*Y*L6x2 - (7-1)/7.*L5x2 #26.8125*Y**7 - 43.3125*Y**5 + 19.6875*Y**3 - 2.1875*Y
    L8x2 = (2*8-1)/8.*Y*L7x2 - (8-1)/8.*L6x2 #50.2734375*Y**8 - 93.84375*Y**6 + 54.140625*Y**4 - 9.84375*Y**2 + 35/128*ones
#   print(L1x2)
#   print(L8x2)
#   print(w)
#   w = np.ones(45)
#   w = 5.*w
    Z = w[0]*L0 + w[1]*L1x1 + w[2]*L1x2 + w[3]*L2x1 + w[4]*L1x1*L1x2 + w[5]*L2x2 + w[6]*L3x1 + \
            w[7]*L2x1*L1x2 + w[8]*L1x1*L2x2 + w[9]*L3x2 + w[10]*L4x1 + w[11]*L3x1*L1x2 + \
            w[12]*L2x1*L2x2 + w[13]*L1x1*L3x2 + w[14]*L4x2 + w[15]*L5x1 + w[16]*L4x1*L1x2 + \
            w[17]*L3x1*L2x2 + w[18]*L2x1*L3x2 + w[19]*L1x1*L4x2 + w[20]*L5x2 + w[21]*L6x1 + \
            w[22]*L5x1*L1x2 + w[23]*L4x1*L2x2 + w[24]*L3x1*L3x2 + w[25]*L2x1*L4x2 + \
            w[26]*L1x1*L5x2 + w[27]*L6x2 + w[28]*L7x1 + w[29]*L6x1*L1x2 + w[30]*L5x1*L2x2 + \
            w[31]*L4x1*L3x2 + w[32]*L3x1*L4x2 + w[33]*L2x1*L5x2 + w[34]*L1x1*L6x2 + w[35]*L7x2 + \
            w[36]*L8x1 + w[37]*L7x1*L1x2 + w[38]*L6x1*L2x2 + w[39]*L5x1*L3x2 + w[40]*L4x1*L4x2 + \
            w[41]*L3x1*L5x2 + w[42]*L2x1*L6x2 + w[43]*L1x1*L7x2 + w[44]*L8x2
    print(Z)
    fig, ax = plt.subplots()
    on = ax.scatter(I_ones, S_ones,s=60, facecolors='none', edgecolors='b')
    fi = ax.scatter(I_fives, S_fives, marker='x', c='r')
    ax.contour(X, Y, Z, [0.0])
    plt.show()

#L0 1
#L1 X
#L2 .5*(3*X**2 - 1)
#L3 .5*(5*X**3 - 3*X)
#L4 .125*(35*X**4 - 30*X**2 + 3)
#L5 .125*(63*X**5 - 70*X**3 + 15*X)
#L6 .0625*(231*X**6 - 315*X**4 + 105*X**2 - 5)
#L7 .0625*(429*X**7 - 693*X**5 + 315*X**3 - 35*X)
#L8 .0078125*(6435*X**8 - 12012*X**6 + 6930*X**4 - 1260*X**2 + 35)

if __name__=="__main__":
    I_ones, S_ones, I_fives, S_fives = get_ones_fives("./zip.all")
    X, Y, Ios, Sos, Ifs, Sfs = get_D(I_ones, S_ones, I_fives, S_fives)
    Z = np.matrix([x for x in X])
    Z_T = Z.getT()
    rows, cols = Z.shape
    I = np.eye(cols)
    y = np.matrix([Y[i] for i in range(len(Y))])
    y = y.getT()
    lamb = 0.0
    wreg0 = get_w(Z, Z_T, I, y, lamb)
    plot(wreg0, Ios, Sos, Ifs, Sfs)









#   fig, ax = plt.subplots()
#   on = ax.scatter(I_ones, S_ones,s=60, facecolors='none', edgecolors='b')
#   fi = ax.scatter(I_fives, S_fives, marker='x', c='r')
#   plt.show()
