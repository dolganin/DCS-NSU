import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import *
from dsp import draw_plot, convolution_mult, convolution_fft, \
                gaussian_kernel, signal_generator
from scipy.stats import norm

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


def bandpass_normal_filter(signal, freq_low, freq_high, discrete_freq):
    sigma = (freq_low+freq_high)/4
    loc = (freq_low+freq_high)/2

    prop = (freq_low+freq_high)/np.pi
    print(sigma, loc)

    norm_samples = np.linspace(-1*prop*loc-5*loc, prop*loc, discrete_freq)
    norm_distribution = norm.pdf(norm_samples, sigma, loc)
    return norm_distribution
def task_5():
    discrete_freq = 300
    x = []
    plt.plot(bandpass_normal_filter(x, 50, 100, discrete_freq))
    plt.xlim(xmin=120, xmax=190)
    plt.show()

    """frequencies = [10, 25, 40]
    signal = signal_generator(frequencies, time)

    spectrum = np.fft.fft(signal)

    filtred_spectrum = spectrum*norm_distribution

    filtred_signal = np.fft.ifft(filtred_spectrum)

    signals = [signal, filtred_signal]
    spectrums = [spectrum, filtred_spectrum]

    fftfreq = np.fft.fftfreq(len(time), 1 / discrete_freq)

    draw_plot(signals, spectrums, fftfreq, num_signals=2, spect_lims_y=True, ylims_spectrum=[0, 20], spect_lims_x=True,
              xlims_spectrum=[-100, 100])"""




task_5()