import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.stats import norm


def draw_plot(signals, spectrums, fftfreq=None, num_signals=1, xlims_signal=None, ylims_signal=None,
              xlims_spectrum=None, ylims_spectrum=None, bars=True):
    """
    This function is motivated by permanent neccesity to dispaly graphics in DSP. These functional part
    is... must to be upgraded.

    :param signals - list of signals which you want to display: list
    :param spectrums - list of spectrums of these signals: list
    :param fftfreq - frequency to display height of spectrum (NB: there are only one value: if you need to display
           different spectrums, so you need to... write another function (?): float
    :param num_signals - num of signals, which need to plot: int
    :param xlims_signal - limit on x axis at the signal: bool
    :param ylims_signal - limit on y axis at the signal: bool
    :param xlims_spectrum - limit on x axis at the spectrum: bool
    :param ylims_spectrum - limit on y axis at the spectrum: bool
    :param bars - if true then it will make bars else it will be plot: bool
    :return: displays signals in the opposite of their spectrums: None
    """
    signals = np.array(signals)
    spectrums = np.array(spectrums)

    signals = signals.reshape(num_signals, -1)  # We need to reshape these lists to make a view, where every signal
    # opposites their spectrum.
    spectrums = spectrums.reshape(num_signals, -1)

    lght = len(signals)
    plt.figure(figsize=(12, 5 * lght))  # Size of the figure depends on the length of signals. If we want to
    # make a "tight" fit, so let's calculate sizes with our data

    for i in range(lght):
        plt.subplot(lght, 2, (2 * i) + 1)  # There we work on every even subplot.
        plt.plot(signals[i])  # First, we plot i-th signal in list.
        if xlims_signal is None:
            plt.xlim(xlims_signal[0], xlims_signal[1])  # Then we need to set x limits at this plot.
        if ylims_signal is None:
            plt.ylim(ylims_signal[0], ylims_signal[1])  # Then we need to set x limits at this plot.
            # Notice: you need to understand that scale of a subplot commot to every subplot.

        plt.subplot(lght, 2, (2 * i) + 2)  # And there we work on every odd subplot.
        if bars:  # IMO spectrum in bar looks more carefully. But we haven't to use
                  # always - you can choose this parameter.
            plt.bar(fftfreq, np.abs(spectrums[i]))
        elif fftfreq is None:  # Same to the spectrums as the signals
            plt.plot(spectrums[i])
        else:
            plt.plot(fftfreq, np.abs(spectrums[i]))

        if xlims_spectrum is None:
            plt.xlim(xlims_spectrum[0], xlims_spectrum[1])
        if ylims_spectrum is None:
            plt.ylim(ylims_spectrum[0], ylims_spectrum[1])

    plt.tight_layout()  # These will automatically relocate subplots to reach the most effectivenely location.
    plt.show()
    return None


def chebyshev_filter(order, cutoff_freq, signal,  ripple=1):
    """
    Naive implementation of Chebyshev filter with the first formula I've found in google. The main specific of this
    filter is that more steep cut on the AFD and there is a more wave-form in the signal before cut
    :param order: this is a parameter of calculation's order(e.g. if order=20, then we calculate from 1 to 21 for
    every sample): int
    :param ripple: there is a waveness which we need to implement in our signal, must be greater or equals zero, in default equals 1: float
    :param cutoff_freq: there is a frequency where we are cut input signal (e.g. if we use signal with harmonics such as
    [1,2,4] and cutoff_freq = 3, then we recieve a signal with harmonics [1, 2]: float
    :param signal: signal which we need to filtrate: list
    :return: signal after application filter: list
    """

    epsilon = np.sqrt(10 ** (0.1 * ripple) - 1)  # Notice: if ripple < 0 then we caught exception because of even
                                                 # root from negative.
    v = np.arccosh((cmath.sqrt(10 ** (-0.1 * ripple) - 1)) / epsilon)  # Poles in our task have to be complex value
                                                                       # cmath calculates complex values.
    u = np.sinh(v / order)   # Calculate the poles of our filter.
    a = np.sinh(v) / (order * u)  # Notice: poles are points where our transfer function aspires to infinity.

    omega_cutoff = np.tan(np.pi * cutoff_freq)
    z = np.exp(-1j * np.pi * np.arange(order + 1) / order)  # We will work with imaginary numbers,
                                                            # so we took them in exponential form

    s = omega_cutoff * (z + 1) / (omega_cutoff * (z - 1) + epsilon)

    filtered_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        for j in range(1, order + 1):
            filtered_signal[i] += (a * np.abs(s[j]) ** 2) / (1 + a * np.abs(s[j]) ** 2) * signal[i]
            signal[i] = (2 * a * np.real(s[j])) / (1 + a * np.abs(s[j]) ** 2) * signal[i]

    return filtered_signal


def butterworth_filter(signal, cutoff_freq, fs, order):
    """
    Naive implementation of Butterworth filter.
    :param signal: original signal: list
    :param cutoff_freq: there is a frequency where we are cut input signal (e.g. if we use signal with harmonics such as
    [1,2,4] and cutoff_freq = 3, then we recieve a signal with harmonics [1, 2]: float
    :param fs: sampling rate of input signal (Notice: it equals lenght of list-signal): int
    :param order: this is a parameter of calculation's order(e.g. if order=20, then we calculate from 1 to 21 for
    every sample): int
    :return: signal after application filter: list
    """
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


def butterworth_filter_low(frequencies, cutoff_freq):
    """
    Simple implementation of Butterworth lowpass filter. First of all, you need to decompose your signal into the
    harmonic's list(e.g.:
    signal = np.sin(2*time)+np.sin(4*time)+np.sin(8*time), then
    harmonic_1 = 2
    harmonic_2 = 4
    ...
    And so on.
    Calculates the coefficients for signal's frequencies. You should multiply the result with your signal's harmonics then.
    :param frequencies: list of harmonics: list
    :param cutoff_freq: there is a frequency where we are cut input signal (e.g. if we use signal with harmonics such as
    [1,2,4] and cutoff_freq = 3, then we recieve a signal with harmonics [1, 2]: float
    :return: coefficiens for spectrum of signal: list
    """
    return [cutoff_freq ** 2 / (-freq ** 2j + np.sqrt(2) * cutoff_freq * freq + 1) for freq in frequencies]


def butterworth_filter_high(frequencies, cutoff_freq):
    """
    Simple implementation of Butterworth lowpass filter. First of all, you need to decompose your signal into the
    harmonic's list(e.g.:
    signal = np.sin(2*time)+np.sin(4*time)+np.sin(8*time), then
    harmonic_1 = 2
    harmonic_2 = 4
    ...
    And so on.
    Calculates the coefficients for signal's frequencies. You should multiply the result with your signal's harmonics then.
    :param frequencies: list of harmonics: list
    :param cutoff_freq: there is a frequency where we are cut input signal (e.g. if we use signal with harmonics such as
    [1,2,4] and cutoff_freq = 3, then we recieve a signal with harmonics [4]: float
    :return: coefficiens for spectrum of signal: list
    """
    return [freq ** 2 / (-cutoff_freq ** 2j + np.sqrt(2) * cutoff_freq * freq + 1) for freq in frequencies]


def signal_generator(frequencies, time, summary_signal=True):
    """
    :param frequencies:
    :param time:
    :param summary_signal:
    :return:
    """
    if summary_signal:
        signal = sum([np.cos(frequencies[i] * 2 * np.pi * time) for i in range(len(frequencies))])
    else:
        signal = [np.cos(frequencies[i] * 2 * np.pi * time) for i in range(len(frequencies))]
    return signal


def fast_fourier_transform(x):
    """
    :param x: 
    :return: 
    """
    x = np.asarray(x)
    N = x.shape[0]
    if N <= 1:
        return x
    even = fast_fourier_transform(x[::2])
    odd = fast_fourier_transform(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]


def discrete_fourier_transform(x):
    """

    :param x: 
    :return: 
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def convolution_mult(signal1, signal2):
    """

    :param signal1:
    :param signal2:
    :return:
    """

    length = len(signal1)
    conv = [0] * length

    for i in range(length):
        for j in range(length):
            conv[i] += signal1[j] * signal2[i - j]

    return conv


def convolution_fft(signal1, signal2):
    """

    :param signal1:
    :param signal2:
    :return:
    """
    conv_len = len(signal1)

    signal1_padded = np.pad(signal1, (0, conv_len - len(signal1)), 'constant')
    signal2_padded = np.pad(signal2, (0, conv_len - len(signal2)), 'constant')

    result = np.fft.ifft(np.fft.fft(signal1_padded) * np.fft.fft(signal2_padded))

    return result


def gaussian_kernel(size, sigma):
    """

    :param size:
    :param sigma:
    :return:
    """
    offset = size // 2
    kernel = np.zeros((size, size))
    for x in range(-offset, offset + 1):
        for y in range(-offset, offset + 1):
            kernel[x + offset, y + offset] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def bandpass_normal_filter(signal, freq_low, freq_high, discrete_freq):
    """

    :param signal:
    :param freq_low:
    :param freq_high:
    :param discrete_freq:
    :return:
    """
    freqs = np.fft.fftfreq(len(signal), 1 / discrete_freq)

    mu = (freq_low + freq_high) / 2
    sigma = (freq_high - freq_low) / 4
    pdf = norm.pdf(freqs, mu, sigma)
    spectrum = np.abs(np.fft.fft(signal))

    filtered_spectrum = spectrum * pdf

    mask = int(np.max(spectrum) + np.min(spectrum) / np.max(filtered_spectrum))

    filtered_spectrum *= mask
    filtered_signal = np.fft.ifft(filtered_spectrum)

    return filtered_signal, filtered_spectrum, spectrum, freqs


def low_pass_filter(signal, cutoff_freq, discrete_freq):
    """

    :param signal:
    :param cutoff_freq:
    :param discrete_freq:
    :return:
    """
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
    return eps * (N - 1) * (1 / k + 1 / (k - eps * (N - 1)))


def zb(k, eps, N):
    return eps * (N - 1) * (1 / (N - 1 - k) + 1 / (-k + (1 - eps) * (N - 1)))


def a(k, eps, N):
    if k == 0 or k == N - 1:
        return 0
    elif k < eps * (N - 1):
        return 1 / (np.exp(za(k, eps, N)) + 1)
    elif k <= (1 - eps) * (N - 1):
        return 1
    elif k < N - 1:
        return 1 / (np.exp(zb(k, eps, N)) + 1)


def plank(eps, spectrum, N, low_freq, high_freq):
    plank = [a(k, eps, high_freq) for k in range(low_freq, high_freq)]
    for i in range(N - high_freq):
        plank.append(0)

    temp_plank = np.zeros(len(spectrum))
    temp_plank[low_freq:len(spectrum)] = plank
    plank = temp_plank

    filtred_spectrum = spectrum * plank
    filtred_signal = np.fft.ifft(filtred_spectrum)

    return filtred_signal, filtred_spectrum


def morle_wavelet(omega, time, alpha):
    return np.exp(-time ** 2 / alpha ** 2) * np.exp(2j * np.pi * time), alpha * np.sqrt(np.pi) * np.exp(
        -alpha ** 2 * (2 * np.pi - omega) ** 2 / 4)


def mexico_hat_wavelet(time):
    return (1 - time ** 2) * np.exp(-time ** 2 / 2)


def haare(x):
    if x >= 0 and x < 1 / 2:
        x = 1
    elif x >= 1 / 2 and x < 1:
        x = -1
    else:
        x = 0
    return x


def haare_wavelet(time):
    return list(map(haare, time))
