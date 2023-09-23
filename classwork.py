import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import nquad

def sin_taylor(x, order):

    sum = 0
    for n in range(order):
        sum += (-1)**n*x**(2*n+1)/(math.factorial(2*n+1))
    return sum

def exp_taylor(x, order):

    sum = 0
    for n in range(order):
        sum += x**n/math.factorial(n)
    return sum

def main():

    order_3 = 3
    order_5 = 5
    order_7 = 7
    order_10 = 10
    order_50 = 50

    x_lst = np.linspace(0, 20, 1000)

    y_lst_sin = np.sin(x_lst)
    y_lst_exp = np.exp(x_lst)

    y_lst_sin_approx_3 = np.zeros(len(x_lst))
    y_lst_sin_approx_5 = np.zeros(len(x_lst))
    y_lst_sin_approx_7 = np.zeros(len(x_lst))
    y_lst_sin_approx_10 = np.zeros(len(x_lst))
    y_lst_sin_approx_50 = np.zeros(len(x_lst))

    y_lst_exp_approx_3 = np.zeros(len(x_lst))
    y_lst_exp_approx_5 = np.zeros(len(x_lst))
    y_lst_exp_approx_7 = np.zeros(len(x_lst))
    y_lst_exp_approx_10 = np.zeros(len(x_lst))
    y_lst_exp_approx_50 = np.zeros(len(x_lst))

    for id in range(len(x_lst)):
        y_lst_sin_approx_3[id] = sin_taylor(x_lst[id], order_3)
        y_lst_sin_approx_5[id] = sin_taylor(x_lst[id], order_5)
        y_lst_sin_approx_7[id] = sin_taylor(x_lst[id], order_7)
        y_lst_sin_approx_10[id] = sin_taylor(x_lst[id], order_10)
        y_lst_sin_approx_50[id] = sin_taylor(x_lst[id], order_50)

        y_lst_exp_approx_3[id] = exp_taylor(x_lst[id], order_3)
        y_lst_exp_approx_5[id] = exp_taylor(x_lst[id], order_5)
        y_lst_exp_approx_7[id] = exp_taylor(x_lst[id], order_7)
        y_lst_exp_approx_10[id] = exp_taylor(x_lst[id], order_10)
        y_lst_exp_approx_50[id] = exp_taylor(x_lst[id], order_50)

    figure, axis = plt.subplots(2)

    axis[0].set_xlabel("F(x) = sin(x)")
    axis[0].plot(x_lst, y_lst_sin,  label="Original function")
    axis[0].plot(x_lst, y_lst_sin_approx_3, label="Function's approximation with Taylor's series for 3 members")
    axis[0].plot(x_lst, y_lst_sin_approx_5, label="Function's approximation with Taylor's series for 5 members")
    axis[0].plot(x_lst, y_lst_sin_approx_7, label="Function's approximation with Taylor's series for 7 members")
    axis[0].plot(x_lst, y_lst_sin_approx_10, label="Function's approximation with Taylor's series for 10 members")
    axis[0].plot(x_lst, y_lst_sin_approx_50, label="Function's approximation with Taylor's series for 50 members")
    axis[0].set_ylim(ymin=-1, ymax=1)
    axis[0].legend()

    axis[1].set_xlabel("F(x) = exp(x)")
    axis[1].plot(x_lst, y_lst_exp_approx_3, label="Function's approximation with Taylor's series for 3 members")
    axis[1].plot(x_lst, y_lst_exp_approx_5, label="Function's approximation with Taylor's series for 5 members")
    axis[1].plot(x_lst, y_lst_exp_approx_7, label="Function's approximation with Taylor's series for 7 members")
    axis[1].plot(x_lst, y_lst_exp_approx_10, label="Function's approximation with Taylor's series for 10 members")
    axis[1].plot(x_lst, y_lst_exp_approx_50, label="Function's approximation with Taylor's series for 50 members")
    axis[1].legend()

    plt.show()

def x_bounds():
    return [0, 1]

def y_bounds(x):
    return [0, np.sqrt(1-x**2)]

def z_bounds(x, y):
    return [0, np.sqrt(1-x**2-y**2)]
def integration():
    irrational_function = lambda x, y, z: (x + y + z) / (np.sqrt(2 * x ** 2 + 4 * y ** 2 + 5 * z ** 2))
    return nquad(irrational_function, [z_bounds, y_bounds, x_bounds])[0]

print(integration())
