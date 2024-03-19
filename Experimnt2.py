
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
import uncertainties as unc
from sympy.abc import *

def get_measurementData(filename, MEASUREMENT_FIRST_VALID_INDEX, MEASUREMENT_LAST_VALID_INDEX):
    df = pd.read_csv(filename, header=None)
    t = df.iloc[MEASUREMENT_FIRST_VALID_INDEX:MEASUREMENT_LAST_VALID_INDEX, 3].to_numpy().astype(float)
    x = df.iloc[MEASUREMENT_FIRST_VALID_INDEX:MEASUREMENT_LAST_VALID_INDEX, 5].to_numpy().astype(float)
    v = df.iloc[MEASUREMENT_FIRST_VALID_INDEX:MEASUREMENT_LAST_VALID_INDEX, 7].to_numpy().astype(float)
    a = df.iloc[MEASUREMENT_FIRST_VALID_INDEX:MEASUREMENT_LAST_VALID_INDEX, 9].to_numpy().astype(float)
    return t, x, v, a

def plot_intervals(filename, MEASUREMENT_FIRST_VALID_INDEX, MEASUREMENT_LAST_VALID_INDEX, number_of_intervals):
        
    t, x, v, a = get_measurementData(filename, MEASUREMENT_FIRST_VALID_INDEX, MEASUREMENT_LAST_VALID_INDEX)

    def f(x,A,B):
        return A+B*x
    param,cov = curve_fit(f,t, v)
    A = param[0]
    B = param[1]
    print(A, B)
    # dv = np.std(v-f(x, A, B),ddof=2)

    # Calculate the total number of data points
    total_points = len(t)

    # Calculate the number of points per interval
    points_per_interval = total_points // number_of_intervals


    # Create 5 separate intervals
    t_intervals = [t[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    x_intervals = [x[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    v_intervals = [v[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    a_intervals = [a[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)] 
    
    plot_trajectory(t_intervals, x_intervals, v_intervals, a_intervals, number_of_intervals)


def plot_trajectory(t_intervals, x_intervals, v_intervals, a_intervals, number_of_intervals):
    pi_1 = []
    pi_2 = []
    for i in range(number_of_intervals):
        t = t_intervals[i]
        x = x_intervals[i]
        v = v_intervals[i]
        a = a_intervals[i]
        a_mean = np.mean(a)

        pi_1.append((x[-1]-x[0]) / (v[0] * (t[-1]-t[0])))
        pi_2.append(a_mean * (t[-1]-t[0]) / v[0])

    #plt.scatter(pi_2, pi_1, color='blue', marker='o')
    #plt.title('Natutueal scale analysis')
    #plt.xlabel('Pi2 (dimentionless)')
    #plt.ylabel('Pi1 (dimentionless)')
    #plt.legend()
    #plt.show()

    #do regression 
    def f(x,A,B):
        return A+B*x
    param,cov = curve_fit(f,pi_2,pi_1)
    print(param)
    print(np.sqrt(np.diag(cov)))
    A = unc.ufloat(param[0],np.sqrt(np.diag(cov))[0])
    B = unc.ufloat(param[1],np.sqrt(np.diag(cov))[1])
    print('A = {}'.format(A))
    print('B = {}'.format(B))
    #print(-A.std_score(1.0))
    #print(-B.std_score(-0.5))

    # cmpute y uncertainties
    dy = np.std(pi_1-f(x,A, B),ddof=2)
    print('dy = {}'.format(dy))

    x_values = np.linspace(min(pi_2),max(pi_2),100)
    plt.figure()
    plt.errorbar(x,y,yerr=dy,fmt='o',capsize=4)
    plt.plot(pi_2,pi_1,'b+')
    plt.plot(x_values,f(x_values,*param),'r--')
    plt.show()



# Example usage
number_of_intervals=8

plot_intervals('./final_data/run1_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 130, MEASUREMENT_LAST_VALID_INDEX = 270, number_of_intervals=number_of_intervals)
plot_intervals('./final_data/run2_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 90, MEASUREMENT_LAST_VALID_INDEX = 240, number_of_intervals=number_of_intervals)
plot_intervals('./final_data/run3_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 70, MEASUREMENT_LAST_VALID_INDEX = 190, number_of_intervals=number_of_intervals)
plot_intervals('./final_data/run4_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 140, MEASUREMENT_LAST_VALID_INDEX = 250, number_of_intervals=number_of_intervals)
plot_intervals('./final_data/run5_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 80, MEASUREMENT_LAST_VALID_INDEX = 200, number_of_intervals=number_of_intervals)

