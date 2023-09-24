import numpy as np
from dsp import signal_generator, morle_wavelet, draw_plot, mexico_hat_wavelet, haare_wavelet
import matplotlib.pyplot as plt

def task_1():
    alpha = 3
    discrete_freq = 300
    omega = np.linspace(-10, 10, discrete_freq)
    time = np.linspace(-5, 5, discrete_freq)
    signal, spectrum = morle_wavelet(omega, time, alpha)
    fftfreq = np.fft.fftfreq(len(time), 1 / discrete_freq)

    draw_plot(signal, spectrum, fftfreq, num_signals=1)

def task_2():
    discrete_freq = 300
    time = np.linspace(-5, 5, discrete_freq)
    fftfreq = np.fft.fftfreq(len(time), 1 / discrete_freq)
    signal = mexico_hat_wavelet(time)
    spectrum = np.fft.fft(signal)
    draw_plot(signal, spectrum, fftfreq, num_signals=1, spect_lims_x=True, xlims_spectrum=[-20, 20])

def task_3():
    discrete_freq = 300
    time = np.linspace(-5, 5, discrete_freq)
    signal = haare_wavelet(time)
    spectrum = np.fft.fft(signal)
    fftfreq = np.fft.fftfreq(len(time), 1 / discrete_freq)

    draw_plot(signal, spectrum, fftfreq, num_signals=1, spect_lims_x=True, xlims_spectrum=[-20, 20], sign_lims_x=True, xlims_signal=[120, 200])

def task_4():
    alpha = 3
    discrete_freq = 300
    omega = np.linspace(-10, 10, discrete_freq)

    time = np.linspace(0, 1, discrete_freq)
    freqs = [2, 4, 8]
    signal = signal_generator(freqs, time)

    signal += np.random.normal(size=signal.shape, loc=30, scale=3) / 10

    mexico_hat = mexico_hat_wavelet(time)
    haare = haare_wavelet(time)
    morle = morle_wavelet(omega, time, alpha)

    conved_signal_hat = np.convolve(signal, mexico_hat, mode='same')
    conved_signal_haare = np.convolve(signal, haare, mode='same')
    conved_signal_morle = np.convolve(signal, morle, mode='same')

    conved_spectrum_hat = np.fft.fft(conved_signal_hat)
    conved_spectrum_haare = np.fft.fft(conved_signal_haare)
    conved_spectrum_morle = np.fft.fft(conved_signal_morle)

    signals = [conved_signal_morle, conved_signal_hat, conved_signal_hat]
    spectrums = [conved_spectrum_morle, conved_spectrum_hat, conved_spectrum_hat]
    fftfreq = np.fft.fftfreq(len(time), 1 / discrete_freq)

    draw_plot(signals, spectrums, fftfreq, num_signals=3)


task_4()
