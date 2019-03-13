import matplotlib.pyplot as plt
import random

def new_array():
    return [0.0 for i in range(1000)]

def flip():
    return random.randint(0,1)

def run_experiment():
    coins = new_array()
    random.seed()

    for coin in range(0,1000):
        for i in range(10):
            coins[coin] += flip()

    v_1 = coins[0] / 10.0
    v_rand = coins[random.randrange(1000)] / 10.0
    v_min = min(coins) / 10.0

    return (v_1, v_rand, v_min)


if __name__ == "__main__":
    v_1_array = []
    v_rand_array = []
    v_min_array = []

    num_trials = 20000
    hmm = num_trials / 100
    for trial in range(0,num_trials):
        v_1, v_rand, v_min = run_experiment()
        v_1_array.append(v_1)
        v_rand_array.append(v_rand)
        v_min_array.append(v_min)
        if ((trial+1) % hmm == 0):
            print("#", end="", flush=True)

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

    axs[0].set_ylabel("Number of occurances")
    axs[1].set_xlabel("Frequency of heads in 10 flips")

    axs[0].hist(v_1_array, bins=11)
    axs[1].hist(v_rand_array, bins=11)
    axs[2].hist(v_min_array, bins=11)

    plt.show()

