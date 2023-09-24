import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import *
from dsp import draw_plot, convolution_mult, convolution_fft, \
                gaussian_kernel, signal_generator, bandpass_normal_filter


def task_1():
    time = np.linspace(0, 10, 3000) #Initializing values for signal proccessing
    signal = square(time)

    kernel_a = 2*np.exp(-time**2)
    kernel_b = 27*time+5

    signal_conv_a = np.convolve(signal, kernel_a, mode='same')
    signal_amplified = signal * 200
    signal_conv_b = np.convolve(signal_amplified, kernel_b, mode='same')

    lst_a = [signal, kernel_a, signal_conv_a]
    lst_b = [signal_amplified, kernel_b, signal_conv_b]

    draw_plot(lst_a, lst_b, num_signals=3, spect_lims_y=True, ylims_spectrum=[0, 400], bars=False)



def task_2():
    time = np.linspace(0, 10, 3000)  # Initializing values for signal proccessing
    signal = square(time)

    kernel_a = 2 * np.exp(-time ** 2)
    kernel_b = 27 * time + 5

    signal_conv_a = np.abs(convolution_mult(signal, kernel_a))
    signal_amplified = signal * 200
    signal_conv_b = np.abs(convolution_mult(signal_amplified, kernel_b))

    lst_a = [signal, kernel_a, signal_conv_a]
    lst_a = np.array(lst_a)
    lst_b = [signal_amplified, kernel_b, signal_conv_b]

    draw_plot(lst_a, lst_b, num_signals=3, spect_lims_y=True, ylims_spectrum=[0, 400], bars=False)


def task_3():
    time = np.linspace(0, 10, 3000)  # Initializing values for signal proccessing
    signal = square(time)

    kernel_a = 2 * np.exp(-time ** 2)
    kernel_b = 27 * time + 5

    signal_conv_a = np.abs(convolution_fft(signal, kernel_a))
    signal_amplified = signal * 200
    signal_conv_b = np.abs(convolution_fft(signal_amplified, kernel_b))

    lst_a = [signal, kernel_a, signal_conv_a]
    lst_a = np.array(lst_a)
    lst_b = [signal_amplified, kernel_b, signal_conv_b]

    draw_plot(lst_a, lst_b, num_signals=3, spect_lims_y=True, ylims_spectrum=[0, 400], bars=False)

def task_4():
    discrete_freq = 100
    time = np.linspace(0, 1, discrete_freq)
    freqs = [2, 4, 8]
    signal = signal_generator(freqs, time)
    spectrum = np.fft.fft(signal)
    fftfreq = np.fft.fftfreq(len(time), 1/discrete_freq)

    kernel_size = 5
    sigma = 10.0
    kernel = gaussian_kernel(kernel_size, sigma)

    blurred_cos_signal = np.convolve(signal, kernel.flatten(), mode='same')
    blurred_cos_spectrum = np.fft.fft(blurred_cos_signal)

    signals = [signal, blurred_cos_signal]
    spectrums = [spectrum, blurred_cos_spectrum]

    draw_plot(signals, spectrums,fftfreq, num_signals=2, spect_lims_y=True, ylims_spectrum=[0, 100], spect_lims_x=True, xlims_spectrum=[-20, 20])



def task_5():
    discrete_freq = 300
    frequencies = [20, 50, 150]

    time = np.linspace(0, 1, discrete_freq)

    signal = signal_generator(frequencies, time)

    filtered_signal, filtred_spectrum, spectrum, fftfreq = bandpass_normal_filter(signal, 10, 30, discrete_freq)
    signals = [signal, filtered_signal]
    spectrums = [spectrum, filtred_spectrum]

    draw_plot(signals, spectrums, fftfreq, num_signals=2)



def task_6():
