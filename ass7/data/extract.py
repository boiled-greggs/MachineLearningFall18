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




if __name__=="__main__":
    f = open("./zip.train")
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

    fig, ax = plt.subplots()
    on = ax.scatter(I_ones, S_ones,s=60, facecolors='none', edgecolors='b')
    fi = ax.scatter(I_fives, S_fives, marker='x', c='r')
    plt.xlabel("Average Intensity of Digit")
    plt.ylabel("Average Reflection Symmetry of Digit")
    plt.legend((on, fi),("Ones", "Fives"))
    plt.show()

