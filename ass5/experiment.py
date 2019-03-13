import matplotlib.pyplot as plt
import numpy as np
import random




def f(x):
    return x*x

def new_D():
    x1 = random.uniform(-1.0, 1.0)
    x2 = random.uniform(-1.0, 1.0)
    y1 = x1*x1
    y2 = x2*x2
    D = [(x1,y1), (x2,y2)]
    return D

def makeline(D):
    a = (D[1][1] - D[0][1]) / (D[1][0] - D[0][0])
    b = D[0][1] - a*D[0][0]
    return a,b

def calc_E_out(a, b):
    x1 = random.uniform(-1.0, 1.0)
    x2 = random.uniform(-1.0, 1.0)
    return .5*(a*x1 + b - f(x1))**2 + .5*(a*x2 + b - f(x2))**2

if __name__ == "__main__":
    random.seed()
    
    slopes = []
    intercepts = []
    E_outs = []
    x = []
    y = []
    N = 10000

    for i in range(N):
        D = new_D()
        x.append(D[0][0])
        x.append(D[1][0])
        y.append(D[0][1])
        y.append(D[1][1])
        slope,intercept = makeline(D)
        slopes.append(slope)
        intercepts.append(intercept)
        E_outs.append(calc_E_out(slope,intercept))

    avg_slope = np.average(slopes)
    avg_int = np.average(intercepts)
    avg_E_out = np.average(E_outs)

    print(avg_slope)
    print(avg_int)
    print(avg_E_out)
    
    biases = []
    for i in range(N):
        biases.append(((avg_slope*x[i] + avg_int) - f(x[i]))**2)
    bias = np.average(biases)
    vars_x = []
    for i in range(N):
        vars_x.append((slopes[i]*x[i] + intercepts[i] - (avg_slope*x[i] + avg_int))**2)
    var = np.average(vars_x)

    print(bias)
    print(var)
