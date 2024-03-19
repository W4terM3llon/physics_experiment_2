
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def load_csv_data(filename):
    df = pd.read_csv(filename, header=None)
    t = df.iloc[100:150, 3].to_numpy().astype(float)
    x = df.iloc[100:150, 5].to_numpy().astype(float)
    v = df.iloc[100:150, 7].to_numpy().astype(float)
    a = df.iloc[100:150, 9].to_numpy().astype(float)
    return t, x, v, a

def plot_trajectory(filename):  

    # a = 9.82 * math.sin(0.09/0.6)

    t, x, v, a = load_csv_data(filename)
    a_mean = np.mean(a)

    pi1 = (x-x[0]) / (v[0] * (t-t[0]))
    pi2 = a_mean * (t-t[0]) / v[0]

    plt.scatter(pi1, pi2, color='blue', marker='o')

    plt.title('Natutueal scale analysis')

    plt.xlabel('Pi1 (dimentionless)')

    plt.ylabel('Pi2 (dimentionless)')

    plt.legend()

    plt.show()



# Example usage

plot_trajectory('run4.csv')
