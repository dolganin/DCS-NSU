from scipy.signal import *
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import datetime
from math import pi
from dsp import fast_fourier_transform, discrete_fourier_transform

def rect_signal(x, ord, peroid, amplitude):
    const = amplitude*4/pi
    omeg = 2*pi/(peroid)
    sum = 0
    for k in range(1, ord, 2):
        sum += const*np.sin(k*omeg*x)/k
    return sum

def time_between():
    frequency1 = 5
    frequency2 = 20

    figure, axis = plt.subplots(2, 3)

    time = np.linspace(0, 4, 1000)
    signal = np.cos(2 * np.pi * frequency1 * time) + np.cos(2 * np.pi * frequency2 * time)
    axis[0][0].plot(signal)
    axis[0][0].set_xlabel("Original signal")
    axis[0][0].set_ylim(ymin=-10, ymax=10)

    start_time_fft = datetime.datetime.now()
    spectrum_fft = fft(signal)
    end_time_fft = (datetime.datetime.now() - start_time_fft).total_seconds()

    start_time_dft = datetime.datetime.now()
    spectrum_dft = discrete_fourier_transform(signal)
    end_time_dft = (datetime.datetime.now() - start_time_dft).total_seconds()

    dft_signal = ifft(spectrum_dft)
    fft_signal = ifft(spectrum_fft)

    axis[1][0].plot(dft_signal)
    axis[1][0].set_xlabel("Signal after dft")
    axis[1][0].set_ylim(ymin=-10, ymax=10)
    axis[1][1].plot(fft_signal)
    axis[1][1].set_ylim(ymin=-10, ymax=10)
    axis[1][1].set_xlabel("Signal after fft")

    t = np.linspace(0, 2 * np.pi, 1000)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

    axis[1][2].plot(x, y, color='red')

    axis[0][1].plot(spectrum_fft, label="Time for fft is = "+(str(end_time_fft)))
    axis[0][2].plot(spectrum_dft, label="Time for dft is = "+(str(end_time_dft)))

    axis[0][1].legend()
    axis[0][2].legend()

    axis[0][1].set_xlabel("Spectrum with FFT")
    axis[0][2].set_xlabel("Spectrum with DFT")
    plt.show()

def dft_with_noise():
    figure, axes = plt.subplots(2, 3)

    x_lst = np.linspace(0, 4, 1000)
    y_lst_square = np.zeros(len(x_lst))
    for id in range(len(x_lst)):
        y_lst_square[id] = rect_signal(x_lst[id], 120, peroid=2, amplitude=2)


    spectrum_fft = fft(y_lst_square)
    spectrum_dft = discrete_fourier_transform(y_lst_square)

    y_lst_square_noise = y_lst_square + np.random.normal(0, 0.1, x_lst.shape)

    spectrum_dft_noise = discrete_fourier_transform(y_lst_square_noise)
    spectrum_fft_noise = fft(y_lst_square_noise)

    signal_noise = ifft(spectrum_fft_noise)

    lst = [y_lst_square, spectrum_fft, spectrum_dft, y_lst_square_noise, spectrum_dft_noise, spectrum_fft_noise, signal_noise]

    axes = axes.ravel()

    for name, ax in zip(lst, axes):
        ax.plot(name, label=name)

    axes.reshape(2, 3)

    plt.show()

def hot_to_do_fft():

    fig, axis = plt.subplots(3)

    time = np.linspace(0, 100, 32768)
    signal = np.cos(2 * np.pi * 50 * time)

    start_time_fft = datetime.datetime.now()
    signal_fft = fft(signal)
    fft_time = datetime.datetime.now()-start_time_fft

    start_time_fft_diy = datetime.datetime.now()
    signal_diy_fft = fast_fourier_transform(signal)
    fft_time_diy = datetime.datetime.now() - start_time_fft_diy

    lst_signals = [signal, signal_fft, signal_diy_fft]

    for sign, ax in zip(lst_signals, axis):
        ax.plot(sign, label = sign)


    axis[0].set_xlim(xmin=0, xmax=50)
    axis[0].set_ylim(ymin=-5, ymax=5)
    axis[1].set_xlim(xmin=4975, xmax=5025)
    axis[2].set_xlim(xmin=4975, xmax=5025)
    axis[1].set_xlabel("Library implementation of fft = "+str(fft_time.total_seconds()))
    axis[2].set_xlabel("My implementation of fft = "+str(fft_time_diy.total_seconds()))

    fig.tight_layout()
    plt.show()

hot_to_do_fft()