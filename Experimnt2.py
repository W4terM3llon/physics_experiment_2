
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit

def get_measurementData(filename):
    MEASUREMENT_FIRST_VALID_INDEX = 50 + 20 # offset by 20 to give it v>0
    MEASUREMENT_LAST_VALID_INDEX = 160
    
    df = pd.read_csv(filename, header=None)
    t = df.iloc[MEASUREMENT_FIRST_VALID_INDEX:MEASUREMENT_LAST_VALID_INDEX, 3].to_numpy().astype(float)
    x = df.iloc[MEASUREMENT_FIRST_VALID_INDEX:MEASUREMENT_LAST_VALID_INDEX, 5].to_numpy().astype(float)
    v = df.iloc[MEASUREMENT_FIRST_VALID_INDEX:MEASUREMENT_LAST_VALID_INDEX, 7].to_numpy().astype(float)
    a = df.iloc[MEASUREMENT_FIRST_VALID_INDEX:MEASUREMENT_LAST_VALID_INDEX, 9].to_numpy().astype(float)
    return t, x, v, a

def plot_intervals(filename):
        
    t, x, v, a = get_measurementData(filename)
    # Calculate the total number of data points
    total_points = len(t)

    # Calculate the number of points per interval
    number_of_intervals = 15
    points_per_interval = total_points // number_of_intervals


    # Create 5 separate intervals
    t_intervals = [t[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    x_intervals = [x[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    v_intervals = [v[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    a_intervals = [a[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)] 
    
    plot_trajectory(t_intervals, x_intervals, v_intervals, a_intervals)


def plot_trajectory(t_intervals, x_intervals, v_intervals, a_intervals):
    
    pi_1 = []
    pi_2 = []
    for i in range(15):
        t = t_intervals[i]
        x = x_intervals[i]
        v = v_intervals[i]
        a = a_intervals[i]
        a_mean = np.mean(a)

        pi_1.append((x[-1]-x[0]) / (v[0] * (t[-1]-t[0])))
        pi_2.append(a_mean * (t[-1]-t[0]) / v[0])

    plt.scatter(pi_2, pi_1, color='blue', marker='o')

    plt.title('Natutueal scale analysis')

    plt.xlabel('Pi2 (dimentionless)')

    plt.ylabel('Pi1 (dimentionless)')

    plt.legend()

    plt.show()

    #do regression 
    
    def f(x,A,B):
        return A+B*x
    param,cov = curve_fit(f,pi_2,pi_1)
    print(param)
    print(np.sqrt(np.diag(cov)))
    import uncertainties as unc
    A = unc.ufloat(param[0],np.sqrt(np.diag(cov))[0])
    B = unc.ufloat(param[1],np.sqrt(np.diag(cov))[1])
    print('A = {}'.format(A))
    print('B = {}'.format(B))
    #print(-A.std_score(1.0))
    #print(-B.std_score(-0.5))

    x_values = np.linspace(min(pi_2),max(pi_2),100)
    plt.figure()
    plt.plot(pi_2,pi_1,'b+')
    plt.plot(x_values,f(x_values,*param),'r--')
    plt.show()




# Example usage

plot_intervals('run4.csv')
