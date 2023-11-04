import numpy as np

from dsp import averaging_signal_linear, signal_generator, averaging_signal_quad, median_filter,\
    spectrum_interpolation, discrease_sampling
import matplotlib.pyplot as plt

dicrete_period = 0.001
def task_1():
    time = np.arange(0, 1, dicrete_period)
    frequencies = [5, 10, 12]
    signal = signal_generator(frequencies=frequencies, time=time, signal_function=np.cos)
    signal_noise = signal + np.random.normal(loc=.2, scale=.3, size=signal.shape)
    signal_cleared = averaging_signal_linear(signal_noise)

    spectrum_noise = np.fft.fft(signal_noise)
    spectrum_cleared = np.fft.fft(signal_cleared)

    spectrum_noise = spectrum_noise/max(spectrum_noise)
    spectrum_cleared = spectrum_cleared/max(spectrum_cleared)

    plt.plot(signal_noise)
    plt.show()

    plt.plot(spectrum_noise)
    plt.xlim(xmin=0, xmax=100)
    plt.show()


    plt.plot(signal_cleared)
    plt.show()

    plt.plot(spectrum_cleared)
    plt.xlim(xmin=0, xmax=100)
    plt.show()




def task_2():

    time = np.arange(0, 1, dicrete_period)
    frequencies = [5, 10, 12]
    signal = signal_generator(frequencies=frequencies, time=time, signal_function=np.cos)
    signal_noise = signal + np.random.normal(loc=.2, scale=.3, size=signal.shape)
    signal_cleared = averaging_signal_quad(signal_noise, time)

    spectrum_noise = np.fft.fft(signal_noise)
    spectrum_cleared = np.fft.fft(signal_cleared)

    spectrum_noise = spectrum_noise / max(spectrum_noise)
    spectrum_cleared = spectrum_cleared / max(spectrum_cleared)

    plt.plot(signal_noise)
    plt.show()

    plt.plot(spectrum_noise)
    plt.xlim(xmin=0, xmax=100)
    plt.show()

    plt.plot(signal_cleared)
    plt.show()

    plt.plot(spectrum_cleared)
    plt.xlim(xmin=0, xmax=100)
    plt.show()

def task_3():
    signal = np.random.normal(loc=1, scale=.3, size=(int(1/dicrete_period)))
    signal = signal/max(signal)

    signal_filtred = averaging_signal_quad(signal, signal)
    plt.plot(signal)
    plt.show()
    plt.plot(signal_filtred)
    plt.show()

def task_4():
    signal = np.random.normal(loc=1, scale=.3, size=(int(1 / dicrete_period)))
    signal = signal / max(signal)
    plt.plot(signal)
    plt.show()

    signal_filtred = median_filter(signal, window_size=10)
    plt.plot(signal_filtred)
    plt.show()

def task_5():
    time = np.arange(0, 5, dicrete_period)
    frequencies = [5, 10, 12]
    window = [1000, 1500]
    signal = signal_generator(frequencies=frequencies, time=time, signal_function=np.sin)

    plt.plot(signal)
    plt.show()

    signal_interpolated = spectrum_interpolation(signal, window)

    plt.plot(signal_interpolated)
    plt.show()

def task_6():
    time = np.arange(0, 1, dicrete_period)
    frequencies = [50, 100]
    signal = signal_generator(frequencies=frequencies, time=time, signal_function=np.cos, summary_signal=False)
    signal_discrete = discrease_sampling(signal, frequencies)

    signal = sum(signal)

    plt.plot(signal)
    plt.show()

    plt.plot(signal_discrete)
    plt.show()


def task_7():
    time = np.arange(0, 1, dicrete_period)
    frequencies = [10, 35, 80]
    signal = signal_generator(frequencies=frequencies, time=time, signal_function=np.cos, summary_signal=False)
    signal_discrete = discrease_sampling(signal, frequencies, sum_signal=False)

    plt.plot(signal_discrete[0])
    plt.plot(signal_discrete[1])
    plt.plot(signal_discrete[2])
    plt.ylim(ymin=min(signal_discrete[1]), ymax=max(signal_discrete[2]))
    plt.xlim(xmin=120, xmax=140)
    plt.show()

task_7()