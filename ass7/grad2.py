import matplotlib.pyplot as plt
import numpy as np


def f(x,y):
    return x**2 + 2*y**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

if __name__=="__main__":
    x0 = [0.1, 1., -.5, -1.]
    y0 = [0.1, 1., -.5, -1.]
    eta = 0.01
    itr = 50
    pi = np.pi
    values = []
    coords = []
    for pt in range(4):
        x = x0[pt]
        y = y0[pt]
        print(x,y)
        for i in range(itr):
            gt = [(2*x + 4*pi*np.cos(2*pi*x)*np.sin(2*pi*y)), \
                    (4*y + 4*pi*np.sin(2*pi*x)*np.cos(2*pi*y))]
            vt = np.array([-1.*eta*gti for gti in gt])
            x += vt[0]
            y += vt[1]
        values.append(f(x,y))
        coords.append((x,y))

    for i in range(4):
        print(coords[i], values[i])
        
#   values.append(f(x,y))
#   itrs.append(50)

#   fig, ax = plt.subplots()
#   ax.plot(itrs, values)
#   ax.set_xlabel("iteration")
#   ax.set_ylabel("function value")
#   plt.title("$\eta = 0.1$")
#   plt.show()
