from scipy.signal import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft

def error_func(function, approximation):
    return function - approximation

period = 400 * np.pi
def fourier_series(x, function, n, order):
    y = np.zeros_like(x)
    for i in range(order):
        a_n = 2 / period * np.trapz(function(x) * np.cos(2 * np.pi * n * x / period), x)
        b_n = 2 / period * np.trapz(function(x) * np.sin(2 * np.pi * n * x / period), x)
        y += a_n * np.cos(2 * np.pi * n * x / period) + b_n * np.sin(2 * np.pi * n * x / period)
        n += 1
    return y

def spectrum(signal):
    return np.fft.ifft(signal)

def plot(function, x, y, step, order, coeff):
    x_lst = np.linspace(x, y, step)

    y_lst = function(x_lst)
    y_lst_approx = fourier_series(x_lst,function, coeff, order)
    y_error = error_func(y_lst, y_lst_approx)

    figure, axis = plt.subplots(2, 3)

    axis[0][0].plot(x_lst, y_lst, label='Original Function')
    axis[0][0].plot(x_lst, y_lst_approx, label='Fourier Series Approximation')

    axis[0][0].set_ylim(ymin=-2, ymax=2)
    axis[0][0].set_xlim(xmin=0, xmax=period/100)

    axis[0][1].plot(x_lst, y_error, label="Error with approximation")

    axis[0][1].set_ylim(ymin=-2, ymax=2)
    axis[0][1].set_xlim(xmin=0, xmax=period/100)

    original_spectrum = spectrum(y_lst)
    approx_spectrum = spectrum(y_lst_approx)

    axis[0][2].plot(np.abs(approx_spectrum))
    axis[0][2].plot(np.abs(original_spectrum))
    axis[0][2].set_ylim(ymin=0, ymax=1)
    axis[0][2].set_xlim(xmin=0, xmax=200)

    axis[0][2].set_xlabel("Frequency, Hz")
    axis[0][2].set_ylabel("Amplicity")

    x_lst_noize = x_lst+np.random.random(len(x_lst))/10
    y_lst_noize = function(x_lst_noize)
    y_lst_approx_noize = fourier_series(x_lst_noize, function, coeff, order)

    axis[1][0].plot(x_lst_noize, y_lst_noize, label='Original Function')
    axis[1][0].plot(x_lst_noize, y_lst_approx_noize, label='Fourier Series Approximation')

    axis[1][0].set_ylim(ymin=-2, ymax=2)
    axis[1][0].set_xlim(xmin=0, xmax=period/100)

    axis[1][1].plot(error_func(y_lst_approx_noize, y_lst_approx), label="Error with noise")
    axis[1][1].set_xlim(xmin=0, xmax=100)

    r = np.sin(4 * x_lst)
    x = r * np.cos(x_lst)
    y = r * np.sin(x_lst)
    axis[1][2].plot(x, y)

    axis[0][0].legend()
    axis[0][1].legend()
    axis[1][1].legend()

    plt.show()

plot(np.cos, 0, period, step = 1000, order = 220, coeff = 1)