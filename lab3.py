import numpy as np
import matplotlib.pyplot as plt
import cmath
from dsp import draw_plot, signal_generator, chebyshev_filter, \
     butterworth_filter, butterworth_filter_high, butterworth_filter_low

def task_1():
    discrete_freq = 1000
    time = np.linspace(0, 1, discrete_freq)
    freqs = [50, 150, 450]
    signal = signal_generator(freqs, time)
    spectrum = np.fft.fft(signal)
    fftfreq = np.fft.fftfreq(len(time), 1/discrete_freq)

    draw_plot(signal, spectrum, fftfreq,  ylims_spectrum=[0, 400])

def task_2():
    time = np.linspace(0, 1, 1000)
    frequencies = [50, 150, 450]
    cut_freq = 100
    signal = signal_generator(frequencies, time, summary_signal=False)
    coefficients = butterworth_filter_low(frequencies, cut_freq)

    filtered_signal = (sum([np.abs(imag)*real for imag, real in zip(coefficients, signal)]))
    signal = sum(signal)

    spectrum = np.fft.fft(signal)
    spectrum_filtred = np.fft.fft(filtered_signal)

    signals = [signal, filtered_signal]
    spectrums = [spectrum, spectrum_filtred]

    fftfreq = np.fft.fftfreq(len(time), 1 / 1000)

    draw_plot(signals, spectrums, fftfreq, num_signals=len(signals))



def task_3():
    time = np.linspace(0, 1, 1000)
    frequencies = [50, 150, 450]
    cut_freq = 150
    signal = signal_generator(frequencies, time, summary_signal=False)
    coefficients = butterworth_filter_high(frequencies, cut_freq)

    filtered_signal = (sum([np.abs(imag) * real for imag, real in zip(coefficients, signal)]))
    signal = sum(signal)

    spectrum = np.fft.fft(signal)
    spectrum_filtred = np.fft.fft(filtered_signal)

    signals = [signal, filtered_signal]
    spectrums = [spectrum, spectrum_filtred]

    fftfreq = np.fft.fftfreq(len(time), 1 / 1000)

    draw_plot(signals, spectrums, fftfreq, num_signals=len(signals))



def task_4():
    time = np.linspace(0, 1, 1000)
    frequencies = [50, 150, 450]
    cut_freq = 150
    signal = signal_generator(frequencies, time, summary_signal=False)

    coefficients_high = np.abs(np.array(butterworth_filter_high(frequencies, cut_freq)))
    coefficients_low = np.abs(np.array(butterworth_filter_low(frequencies, cut_freq)))


    coefficients_threshold = coefficients_high * coefficients_low
    coefficients_block = coefficients_high + coefficients_low


    filtered_signal_threshold = (sum([np.abs(imag) * real for imag, real in zip(coefficients_threshold, signal)]))
    filtered_signal_block = (sum([np.abs(imag) * real for imag, real in zip(coefficients_block, signal)]))
    signal = sum(signal)

    spectrum = np.fft.fft(signal)
    spectrum_filtred_threshold = np.fft.fft(filtered_signal_threshold)
    spectrum_filtred_block = np.fft.fft(filtered_signal_block)

    signals = [signal, filtered_signal_block, filtered_signal_threshold]
    spectrums = [spectrum, spectrum_filtred_block, spectrum_filtred_threshold]

    fftfreq = np.fft.fftfreq(len(time), 1 / 1000)

    draw_plot(signals, spectrums, fftfreq, num_signals=len(signals), xlims_signal=[0, 400],  ylims_spectrum=[0, 500])


def task_5_6():
    time = np.linspace(0, 1, 1000)
    frequencies = [50, 150, 450]
    cut_freq = 100

    signal = sum(signal_generator(frequencies, time, summary_signal=False))
    signal_noise = signal + np.random.laplace(size=signal.shape, scale=0.1)
    signal_filter_3 = np.abs(butterworth_filter(signal=signal, cutoff_freq=cut_freq, fs=1000, order=3))
    signal_filter_4 = np.abs(butterworth_filter(signal=signal, cutoff_freq=cut_freq, fs=1000, order=4))

    signal_filter_3_noise = np.abs(butterworth_filter(signal=signal_noise, cutoff_freq=cut_freq, fs=1000, order=3))
    signal_filter_4_noise = np.abs(butterworth_filter(signal=signal_noise, cutoff_freq=cut_freq, fs=1000, order=4))


    spectrum = np.fft.fft(signal)
    spectrum_3 = np.fft.fft(signal_filter_3)
    spectrum_4 = np.fft.fft(signal_filter_4)

    spectrum_noise = np.fft.fft(signal_noise)
    spectrum_3_noise = np.fft.fft(signal_filter_3_noise)
    spectrum_4_noise = np.fft.fft(signal_filter_4_noise)

    fftfreq = np.fft.fftfreq(len(time), 1 / 1000)

    signals = [signal, signal_filter_3, signal_filter_4, signal_noise, signal_filter_3_noise, signal_filter_4_noise]
    spectrums = [spectrum, spectrum_3, spectrum_4, spectrum_noise, spectrum_3_noise, spectrum_4_noise]

    draw_plot(signals, spectrums, fftfreq, num_signals=len(signals), xlims_signal=[0, 400], ylims_spectrum=[0, 200])

def task_7():

    time = np.linspace(0, 1, 1000)
    frequencies = [0.1, 0.2, 0.4]
    order = 2
    ripple = 1.0
    cut_freq = 0.2
    signal = sum(signal_generator(frequencies, time, summary_signal=False))
    filtred_signal = chebyshev_filter(order=order, ripple=ripple, cutoff_freq=cut_freq, signal=signal)

    fftfreq = np.fft.fftfreq(len(time), 1 /1000)
    spectrum = np.fft.fft(signal)
    filtred_spectrum = np.fft.fft(filtred_signal)

    signals = [signal, filtred_signal]
    spectrum = [spectrum, filtred_spectrum]

    draw_plot(signals, spectrum, fftfreq, num_signals=len(signals), xlims_spectrum=[-50, 50])

