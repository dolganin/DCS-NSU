import numpy as np
from dsp import signal_generator, morle_wavelet, draw_plot, mexico_hat_wavelet, haare_wavelet

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
    draw_plot(signal, spectrum, fftfreq, num_signals=1, xlims_spectrum=[-20, 20])

def task_3():
    discrete_freq = 300
    time = np.linspace(-5, 5, discrete_freq)
    signal = haare_wavelet(time)
    spectrum = np.fft.fft(signal)
    fftfreq = np.fft.fftfreq(len(time), 1 / discrete_freq)

    draw_plot(signal, spectrum, fftfreq, num_signals=1, xlims_spectrum=[-20, 20],  xlims_signal=[120, 200])

def task_4():
    alpha = 3
    discrete_freq = 300
    freq_wavelets = 10
    omega = np.linspace(-10, 10, discrete_freq)

    time = np.linspace(0, 1, discrete_freq)

    time_wavelet = np.linspace(-2, 2, freq_wavelets)

    freqs = [2, 4, 8]
    signal = signal_generator(freqs, time)

    signal += np.random.normal(size=signal.shape, loc=30, scale=3) / 10

    mexico_hat = mexico_hat_wavelet(time_wavelet)
    haare = haare_wavelet(time_wavelet)
    morle, _ = morle_wavelet(omega, time_wavelet, alpha)

    conved_signal_hat = np.convolve(signal, mexico_hat, mode='valid')
    conved_signal_haare = np.convolve(signal, haare, mode='valid')
    conved_signal_morle = np.convolve(signal, morle, mode='valid')

    conved_spectrum_hat = np.fft.fft(conved_signal_hat)
    conved_spectrum_haare = np.fft.fft(conved_signal_haare)
    conved_spectrum_morle = np.fft.fft(conved_signal_morle)

    signals = [conved_signal_morle, conved_signal_hat, conved_signal_haare]
    spectrums = [conved_spectrum_morle, conved_spectrum_hat, conved_spectrum_haare]
    fftfreq = np.fft.fftfreq(len(conved_spectrum_morle), 1 / conved_spectrum_morle)

    draw_plot(signals, spectrums, fftfreq, num_signals=3, xlims_spectrum=[-2.5, 2.5], ylims_spectrum=[0, 200])


def task_5():
    pass

