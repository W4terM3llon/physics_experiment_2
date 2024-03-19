
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

    a_mean = np.mean(a)
    dv, da,dx = getErrors(t, x, v, a)
    #propagate_errors(t, x, v, a, dv, da, dx, a_mean)

    # Calculate the total number of data points
    total_points = len(t)
    # Calculate the number of points per interval
    points_per_interval = total_points // number_of_intervals
    # Create 5 separate intervals
    t_intervals = [t[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    x_intervals = [x[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    v_intervals = [v[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)]
    a_intervals = [a[i * points_per_interval: (i + 1) * points_per_interval] for i in range(number_of_intervals)] 
    
    plot_trajectory(t_intervals, x_intervals, v_intervals, a_intervals, number_of_intervals, dv, da, dx, a_mean)

def getErrors(t, x, v, a):
    def f_linear(x,A,B):
        return A+B*x
        
    def f_parabolic(x,A,B,C):
        return A+B*x+C*x**2


    # calculate errors and plot them
    # v by t    
    param,cov = curve_fit(f_linear,t, v)
    A = param[0]
    B = param[1]
    dv = np.std(v-f_linear(t, A, B),ddof=2)
    plt.figure()
    plt.errorbar(t,v,yerr=dv,fmt='o',capsize=4)
    t_values = np.linspace(t[0],t[-1],100)
    plt.plot(t_values,f_linear(t_values, A, B),'b--')
    plt.show()

    # a by t
    param,cov = curve_fit(f_linear,t, a)
    A = param[0]
    B = param[1]
    da = np.std(a-f_linear(t, A, B),ddof=2)
    plt.figure()
    plt.errorbar(t,a,yerr=da,fmt='o',capsize=4)
    t_values = np.linspace(t[0],t[-1],100)
    plt.plot(t_values,f_linear(t_values, A, B),'b--')
    plt.show()

    # x by t
    param,cov = curve_fit(f_parabolic,t, x)
    A = param[0]
    B = param[1]
    C = param[2]
    dx = np.std(x-f_parabolic(t, A, B, C),ddof=3)
    plt.figure()
    plt.errorbar(t,x,yerr=dx,fmt='o',capsize=4)
    t_values = np.linspace(t[0],t[-1],100)
    plt.plot(t_values,f_parabolic(t_values, A, B, C),'b--')
    plt.show()

    return dv, da,dx

def propagate_errors_to_calculate_pi(t, x, v, a, dv, da, dx, a_mean):
    t_initial_error = unc.ufloat(t[0], 0)
    t_final_error = unc.ufloat(t[-1], 0)
    x_initial_error = unc.ufloat(x[0], dx)
    x_final_error = unc.ufloat(x[-1], dx)
    v_initial_error = unc.ufloat(v[0], dv)
    a_mean_error = unc.ufloat(a_mean, 0)
    

    x_error = [x_initial_error, x_final_error]
    t_error = [t_initial_error, t_final_error]
    v_error = [v_initial_error]

    pi_1 = calculate_pi_1(x_error, v_error, t_error)
    pi_2 = calculate_pi_2(a_mean_error, v_error, t_error)
    print(pi_1, pi_2) # print pi errors

    return (pi, pi_2)
    
def calculate_pi_1(x, v, t):
    return (x[-1]-x[0]) / (v[0] * (t[-1]-t[0]))

def calculate_pi_2(a_mean, v, t):
    return a_mean * (t[-1]-t[0]) / v[0]

def plot_trajectory(t_intervals, x_intervals, v_intervals, a_intervals, number_of_intervals, dv, da, dx, a_mean):
    pi_1 = []
    pi_2 = []
    for i in range(number_of_intervals):
        t = t_intervals[i]
        x = x_intervals[i]
        v = v_intervals[i]
        a = a_intervals[i]
        #a_mean = np.mean(a)

        propagate_errors_to_calculate_pi(t,x,v,a,dv,da,dx,a_mean)
        pi_1.append(calculate_pi_1(x, v, t))
        pi_2.append(calculate_pi_2(a_mean, v, t))

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
    A = unc.ufloat(param[0],np.sqrt(np.diag(cov))[0])
    B = unc.ufloat(param[1],np.sqrt(np.diag(cov))[1])
    print('A = {}'.format(A))
    print('B = {}'.format(B))
    #print(-A.std_score(1.0))
    #print(-B.std_score(-0.5))

    plt.figure()
    plt.plot(pi_2,pi_1,'b+')
    plt.show()



# Example usage
number_of_intervals=8

plot_intervals('./final_data/run1_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 130, MEASUREMENT_LAST_VALID_INDEX = 270, number_of_intervals=number_of_intervals)
plot_intervals('./final_data/run2_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 90, MEASUREMENT_LAST_VALID_INDEX = 240, number_of_intervals=number_of_intervals)
plot_intervals('./final_data/run3_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 70, MEASUREMENT_LAST_VALID_INDEX = 190, number_of_intervals=number_of_intervals)
plot_intervals('./final_data/run4_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 140, MEASUREMENT_LAST_VALID_INDEX = 250, number_of_intervals=number_of_intervals)
plot_intervals('./final_data/run5_height_4.csv', MEASUREMENT_FIRST_VALID_INDEX = 80, MEASUREMENT_LAST_VALID_INDEX = 200, number_of_intervals=number_of_intervals)

