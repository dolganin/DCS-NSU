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
    nyquist_freq = fs / 2  # By the Kotelknikov's theorem.
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    poles = []  # To this list we will append poles of Butterworth filter.
    for k in range(1, order + 1):
        real = -np.cos((2 * k + order - 1) * np.pi / (2 * order))  # Generate coefficients for poles
        imag = np.sin((2 * k + order - 1) * np.pi / (2 * order))  # to work with complex values.
        poles.append(complex(real, imag))
    normalized_poles = []
    for pole in poles:
        normalized_poles.append(pole * normalized_cutoff_freq)  # Normalize poles with using normalized frequency
    filtered_signal = np.zeros(len(signal))
    for i in range(len(signal)):  # For every element from original signal we copy them to zeros-array and
        # then -/+ real and imaginary part of calculations.
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


def signal_generator(frequencies, time, summary_signal=True, signal_function=np.cos):
    """
    Because of permanently using of signal_generating, implement this function.
    :param frequencies: harmonics of your signal: list
    :param time: linspace to apply functions with harmonics
    :param summary_signal: if False then we will recieve n lists of signal, else only one list: [[]]
    :param signal_function: you may take a signal with any functions. In default = cos, but you can cnage, for exmaple,
    sin.
    :return: list of lists or list, representing a signal.
    """
    if summary_signal:  # If true then there is only one list.
        signal = sum([signal_function(frequencies[i] * 2 * np.pi * time) for i in range(len(frequencies))])  # Generate
        # sequence of signals with different harmonics and summarize it.
    else:  # Else we gain list of lists.
        signal = [signal_function(frequencies[i] * 2 * np.pi * time) for i in range(len(frequencies))]
    return signal


def fast_fourier_transform(x):
    """
    Standard algorithm FFT.
    Notice: work only on sequences, which lenght equals one of the power of 2 (e.g. 2, 4, 8...1024...)
    This trick is based on fact that Spectrum is symmetrical and we can calculate only every 2-th frequency there to
    get spectrum.
    :param x: original signal: list
    :return: spectrum of signal
    """
    x = np.asarray(x)  # We will work with x as array (it can be list).
    N = x.shape[0]
    if N <= 1:
        return x
    even = fast_fourier_transform(x[::2])  # Recursive call for evens and odds.
    odd = fast_fourier_transform(x[1::2])  #
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]


def discrete_fourier_transform(x):
    """
    Naive implementaion of Fourier Transform in values.
    :param x: original signal: list
    :return: spectrum of signal
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def convolution_mult(signal1, signal2):
    """
    Intuitive algorithm of mixing two signals.
    Notice: signal1 and signal2 must be one shape.
    :param signal1: list
    :param signal2: list
    :return: conved signal
    """

    conv = np.zeros(len(signal1))  # Initialize zeros-array with length equals to signals.

    for i in range(len(signal1)):
        for j in range(len(signal1)):
            conv[i] += signal1[j] * signal2[i - j]  # Naive and 1-dimensional convolution.
    return conv


def convolution_fft(signal1, signal2):
    """
    Because of theorem about FFT, we gain that convolution of two signals in time domain equals to
    multiply in frequency domain
    :param signal1: list
    :param signal2: list
    :return: conved_signal
    """
    conv_len = len(signal1)

    signal1_padded = np.pad(signal1, (0, conv_len - len(signal1)), 'constant')
    signal2_padded = np.pad(signal2, (0, conv_len - len(signal2)), 'constant')

    result = np.fft.ifft(np.fft.fft(signal1_padded) * np.fft.fft(signal2_padded))

    return result


def gaussian_kernel(size, sigma):
    """
    Implementation of gaussian's kernel for CV, but it is only 1-dimensional. Performing a blurring
    above the signal.
    :param size: size of the kernel: int
    :param sigma: coefficient of blurring: float
    :return: kernel for blurring: list
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
    There is a intresting idea: let's use a normal distribution to filter signal. How? We can define spread of
    these distribution with parameter root-mean-square deviation and the middle of distributuion is on the expectation.
    Then we multiply these with spectrum of our signal and perform the inverse fast fourier transform.
    :param signal: original signal: list
    :param freq_low: this is a frequency where our filter starts: float
    :param freq_high: and there our filter ends: float
    :param discrete_freq: sampling rate of input signal (Notice: it equals lenght of list-signal): int
    :return: signal after performing filtering, spectrum after performing filtering, spectrum of original signal and frequency
    to define height of bars on the plot for spectrum
    """
    freqs = np.fft.fftfreq(len(signal), 1 / discrete_freq)  # Calculate frequency to plot heights.

    mu = (freq_low + freq_high) / 2  # Calculate expecation in the following consideration
    sigma = (freq_high - freq_low) / 4  # Calcultate the root-mean-square deviation.
    pdf = norm.pdf(freqs, mu, sigma)  # Build the distribution.
    spectrum = np.abs(np.fft.fft(signal))

    filtered_spectrum = spectrum * pdf

    mask = int(np.max(spectrum) + np.min(spectrum) / np.max(filtered_spectrum))  # This is variale of scaling. Our
    # signal will be decreased by multiplying with normal distribution, so we need to increase it backward.

    filtered_spectrum *= mask
    filtered_signal = np.fft.ifft(filtered_spectrum)

    return filtered_signal, filtered_spectrum, spectrum, freqs


def low_pass_filter(signal, cutoff_freq, discrete_freq):
    """
    Inspired by bandpass_normal_filter (look the upper function). If we use as lower bound zero and cutoff_freq as
    upper bound, we clip the signal in these points.
    :param signal: list
    :param cutoff_freq: there is a frequency where we are cut input signal (e.g. if we use signal with harmonics such as
    [1,2,4] and cutoff_freq = 3, then we recieve a signal with harmonics [1, 2]: float
    :param discrete_freq: sampling rate of input signal (Notice: it equals lenght of list-signal): int
    :return: filtred signal without harmonics >= cutoff_freq
    """
    return bandpass_normal_filter(signal, 0, cutoff_freq, discrete_freq)

def plank(eps, spectrum, N, low_freq, high_freq):
    """
    It's something inside you... It's hard to explain...



    :param eps:
    :param spectrum:
    :param N:
    :param low_freq:
    :param high_freq:
    :return:
    """
    def coefficient_a(k, eps, N):
        if k == 0 or k == N - 1:
            return 0
        elif k < eps * (N - 1):
            return 1 / (np.exp(coefficient_za(k, eps, N)) + 1)
        elif k <= (1 - eps) * (N - 1):
            return 1
        elif k < N - 1:
            return 1 / (np.exp(coefficient_zb(k, eps, N)) + 1)

    def coefficient_za(k, eps, N):
        return eps * (N - 1) * (1 / k + 1 / (k - eps * (N - 1)))

    def coefficient_zb(k, eps, N):
        return eps * (N - 1) * (1 / (N - 1 - k) + 1 / (-k + (1 - eps) * (N - 1)))


    plank = [coefficient_a(k, eps, high_freq) for k in range(low_freq, high_freq)]
    for i in range(N - high_freq):
        plank.append(0)

    temp_plank = np.zeros(len(spectrum))
    temp_plank[low_freq:len(spectrum)] = plank
    plank = temp_plank

    filtred_spectrum = spectrum * plank
    filtred_signal = np.fft.ifft(filtred_spectrum)

    return filtred_signal, filtred_spectrum


def morle_wavelet(omega, time, alpha):
    """
    Naive implementation of Morle's wavelet.
    :param omega: signal's harmonic's list: list
    :param time: list of values: list
    :param alpha: coefficient of scaling: float
    :return: filtred_signal, filtred_spectrum
    """
    return np.exp(-time ** 2 / alpha ** 2) * np.exp(2j * np.pi * time), alpha * np.sqrt(np.pi) * np.exp(
        -alpha ** 2 * (2 * np.pi - omega) ** 2 / 4)


def mexico_hat_wavelet(time):
    """
              .~
              ^^.
              ^.:
             .: ^
..........   :. ^.  ..........
..........:. ^  :: ::.........
           ::^   ^^:
            .    ..
    :param time: list
    :return: result: list
    """
    return (1 - time ** 2) * np.exp(-time ** 2 / 2)


def haare(x):
    """
    :::::::::::.
    :         .:
    :          :
    :          :
    :          :
....:          :          ....
.....          :         .:...
               :         ..
               :         ..
               :         ..
               :         .:
               :::::::::::.
    :param x: value: float
    :return: result: float
    """
    if x >= 0 and x < 1 / 2:
        x = 1
    elif x >= 1 / 2 and x < 1:
        x = -1
    else:
        x = 0
    return x


def haare_wavelet(time):
    return list(map(haare, time))
