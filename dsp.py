import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.stats import norm
from scipy.signal import find_peaks

def draw_plot(signals, spectrums, fftfreq=None, num_signals=1, xlims_signal=None, ylims_signal=None,
              xlims_spectrum=None, ylims_spectrum=None, sign_lims_x=False, spect_lims_x=False,
              sign_lims_y=False, spect_lims_y=False, bars=True):
    signals = np.array(signals)
    spectrums = np.array(spectrums)

    signals = signals.reshape(num_signals, -1)
    spectrums = spectrums.reshape(num_signals, -1)

    lght = len(signals)
    plt.figure(figsize=(12, 5 * lght))

    for i in range(lght):
        plt.subplot(lght, 2, (2 * i) + 1)
        plt.plot(signals[i])
        if sign_lims_x:
            plt.xlim(xlims_signal[0], xlims_signal[1])
        if sign_lims_y:
            plt.ylim(ylims_signal[0], ylims_signal[1])

        plt.subplot(lght, 2, (2 * i) + 2)
        if(bars):
            plt.bar(fftfreq, np.abs(spectrums[i]))
        elif (fftfreq == None):
            plt.plot(spectrums[i])
        else:
            plt.plot(fftfreq, np.abs(spectrums[i]))

        if spect_lims_x:
            plt.xlim(xlims_spectrum[0], xlims_spectrum[1])
        if spect_lims_y:
            plt.ylim(ylims_spectrum[0], ylims_spectrum[1])

    plt.tight_layout()
    plt.show()


def chebyshev_filter(order, ripple, cutoff_freq, signal):
    # Вычисление параметров фильтра
    epsilon = np.sqrt(10 ** (0.1 * ripple) - 1)
    v = np.arccosh((cmath.sqrt(10 ** (-0.1 * ripple) - 1)) / epsilon)
    u = np.sinh(v / order)
    a = np.sinh(v) / (order * u)

    # Препарирование фильтра
    omega_cutoff = np.tan(np.pi * cutoff_freq)
    z = np.exp(-1j * np.pi * np.arange(order + 1) / order)
    s = omega_cutoff * (z + 1) / (omega_cutoff * (z - 1)+epsilon)

    # Применение фильтра к сигналу
    filtered_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        for j in range(1, order + 1):
            filtered_signal[i] += (a * np.abs(s[j]) ** 2) / (1 + a * np.abs(s[j]) ** 2) * signal[i]
            signal[i] = (2 * a * np.real(s[j])) / (1 + a * np.abs(s[j]) ** 2) * signal[i]

    return filtered_signal


def butterworth_filter(signal, cutoff_freq, fs, order):
    nyquist_freq = fs / 2
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    poles = []
    for k in range(1, order + 1):
        real = -np.cos((2 * k + order - 1) * np.pi / (2 * order))
        imag = np.sin((2 * k + order - 1) * np.pi / (2 * order))
        poles.append(complex(real, imag))
    normalized_poles = []
    for pole in poles:
        normalized_poles.append(pole * normalized_cutoff_freq)
    filtered_signal = [0] * len(signal)
    for i in range(len(signal)):
        filtered_signal[i] = signal[i]
        for pole in normalized_poles:
            filtered_signal[i] -= pole.real * filtered_signal[i - 1].real
            filtered_signal[i] += pole.imag * filtered_signal[i - 1].imag
    return filtered_signal

def butterworth_filter_low(frequencies, cut_freq):
    return [cut_freq**2/(-freq**2j+np.sqrt(2)*cut_freq*freq+1) for freq in frequencies]

def butterworth_filter_high(frequencies, cut_freq):
    return [freq**2/(-cut_freq**2j+np.sqrt(2)*cut_freq*freq+1) for freq in frequencies]

def signal_generator(frequencies, time, summary_signal=True):
    if summary_signal:
        signal = sum([np.cos(frequencies[i] * 2 * np.pi * time) for i in range(len(frequencies))])
    else:
        signal = [np.cos(frequencies[i] * 2 * np.pi * time) for i in range(len(frequencies))]
    return signal


def fast_fourier_transform(x):
    x = np.asarray(x)
    N = x.shape[0]
    if N <= 1:
        return x
    even = fast_fourier_transform(x[::2])
    odd = fast_fourier_transform(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def discrete_fourier_transform(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def convolution_mult(signal1, signal2):
    length = len(signal1)
    conv = [0] * length

    for i in range(length):
        for j in range(length):
            conv[i] += signal1[j] * signal2[i - j]

    return conv


def convolution_fft(signal1, signal2):
    conv_len = len(signal1)

    signal1_padded = np.pad(signal1, (0, conv_len - len(signal1)), 'constant')
    signal2_padded = np.pad(signal2, (0, conv_len - len(signal2)), 'constant')

    result = np.fft.ifft(np.fft.fft(signal1_padded) * np.fft.fft(signal2_padded))

    return result

def gaussian_kernel(size, sigma):
    offset = size // 2
    kernel = np.zeros((size, size))
    for x in range(-offset, offset + 1):
        for y in range(-offset, offset + 1):
            kernel[x + offset, y + offset] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def bandpass_normal_filter(signal, freq_low, freq_high, discrete_freq):
    freqs = np.fft.fftfreq(len(signal), 1 / discrete_freq)

    mu = (freq_low + freq_high) / 2
    sigma = (freq_high - freq_low) / 4
    pdf = norm.pdf(freqs, mu, sigma)
    spectrum = np.abs(np.fft.fft(signal))

    filtered_spectrum = spectrum*pdf

    mask = int(np.max(spectrum)+np.min(spectrum)/np.max(filtered_spectrum))

    filtered_spectrum *= mask
    filtered_signal = np.fft.ifft(filtered_spectrum)

    return filtered_signal, filtered_spectrum, spectrum, freqs

def low_pass_filter(signal, cutoff_freq, discrete_freq):

    freqs = np.fft.fftfreq(len(signal), 1 / discrete_freq)

    spectrum = np.abs(np.fft.fft(signal))

    mu = (cutoff_freq) / 2
    sigma = (cutoff_freq) / 4
    pdf = norm.pdf(freqs, mu, sigma)

    filtered_spectrum = spectrum * pdf

    mask = int(np.max(spectrum) + np.min(spectrum) / np.max(filtered_spectrum))

    filtered_spectrum *= mask

    filtered_signal = np.fft.ifft(filtered_spectrum)

    return filtered_signal, filtered_spectrum, spectrum, freqs

def za(k, eps, N):
    return eps * (N-1) * (1/k + 1/(k - eps*(N-1)))
def zb(k, eps, N):
    return eps * (N-1) * (1/(N-1-k) + 1/(-k + (1-eps)*(N-1)))
def a(k, eps, N):
    if k == 0 or k == N-1:
        return 0
    elif k < eps * (N-1):
        return 1 / (np.exp(za(k, eps, N)) + 1)
    elif k <= (1-eps) * (N-1):
        return 1
    elif k < N-1:
        return 1 / (np.exp(zb(k, eps, N)) + 1)
def plank(eps, spectrum, N, low_freq, high_freq):
    plank = [a(k, eps, high_freq) for k in range(low_freq, high_freq)]
    for i in range(N - high_freq):
        plank.append(0)

    temp_plank = np.zeros(len(spectrum))
    temp_plank[low_freq:len(spectrum)] = plank
    plank = temp_plank


    filtred_spectrum = spectrum*plank
    filtred_signal = np.fft.ifft(filtred_spectrum)

    return filtred_signal, filtred_spectrum

def morle_wavelet(omega, time, alpha):
    return np.exp(-time**2/alpha**2)*np.exp(2j*np.pi*time), alpha*np.sqrt(np.pi)*np.exp(-alpha**2*(2*np.pi-omega)**2/4)

def mexico_hat_wavelet(time):
    return (1-time**2)*np.exp(-time**2/2)

def haare(x):
    if x>=0 and x < 1/2:
        x = 1
    elif x>=1/2 and x<1:
        x = -1
    else:
        x = 0
    return x

def haare_wavelet(time):
    return list(map(haare, time))


