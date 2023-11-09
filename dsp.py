import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.stats import norm
import scipy.integrate as integrate
from scipy.integrate import quad, nquad

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
        if xlims_signal is not None:
            plt.xlim(xlims_signal[0], xlims_signal[1])  # Then we need to set x limits at this plot.
        if ylims_signal is not None:
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

        if xlims_spectrum is not None:
            plt.xlim(xlims_spectrum[0], xlims_spectrum[1])
        if ylims_spectrum is not None:
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
            conv[i] += signal1[j] * signal2[j-i]  # Naive and 1-dimensional convolution.
    return conv


def convolution_fft(signal1, signal2):
    """
    Because of theorem about FFT, we gain that convolution of two signals in time domain equals to
    multiply in frequency domain
    :param signal1: list
    :param signal2: list
    :return: conved_signal
    """
    # Вычисление длины итогового сигнала
    n = len(signal1)
    # Выполнение FFT для сигналов
    fft_signal1 = np.fft.fft(signal1, n)
    fft_signal2 = np.fft.fft(signal2, n)
    # Умножение в частотной области
    fft_result = fft_signal1 * fft_signal2
    # Обратное FFT для получения итогового сигнала
    conv_result = np.real(np.fft.ifft(fft_result))

    return conv_result



def gaussian_kernel(size, sigma):
    """
    Implementation of gaussian's kernel for CV, but it is only 1-dimensional. Performing a blurring
    above the signal.
    :param size: size of the kernel: int
    :param sigma: coefficient of blurring: float
    :return: kernel for blurring: list

      ╱|、
    (˚ˎ 。7
    |、˜〵
    じしˍ,)ノ
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

    This code implements difficult and huge in formulas algortithm of filtering signal. More details algorithm at
    the "return" point.

    :param eps: this parameter defines steep slope of the cut.
    /\      /\
    | eps -> | slope.
    :param spectrum: spectrum of the signal.
    :param N: sampling rate of input signal (Notice: it equals lenght of list-signal): int
    :param low_freq: this is a frequency where our filter starts: float
    :param high_freq: and there our filter ends: float
    :return: list with values according to the next rule: if spectrum[i] in [low_freq, high_freq] then plank[i]=value
    else 0
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


    result_plank = [coefficient_a(k, eps, high_freq) for k in range(low_freq, high_freq)]
    for i in range(N - high_freq):  # We will fill array with zeros to get neccesary length of list.
        result_plank.append(0)

    temp_plank = np.zeros(len(spectrum))  # Also we need to paste some zeros in the beginning of array.
    temp_plank[low_freq:len(spectrum)] = result_plank  # Paste plank coefficients.
    result_plank = temp_plank

    filtred_spectrum = spectrum * result_plank  # Notice: result_plank = [0...0,1...1,0...0], and when we multiply,
    # we got spectrum after filter

    filtred_signal = np.fft.ifft(filtred_spectrum)

    return filtred_signal, filtred_spectrum


def morle_wavelet(omega, time, alpha=1):
    """
    Naive implementation of Morle's wavelet. Just formula.
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


def find_half_max(signal):
    max_amplitude = np.max(signal)
    half_max = max_amplitude / 2
    indices = np.where(signal >= half_max)[0]

    first_index = indices[0]
    last_index = indices[-1]

    if len(indices) % 2 == 0:
        # если количество индексов четное, берем среднее двух соседних индексов для более точного результата
        first_index -= 1
        last_index += 1

    return last_index - first_index


def calc_fwhm(signal):
    first_index, last_index = find_half_max(signal)
    fwhm = last_index - first_index
    return fwhm


def welve(t, alpha=1):
    return np.exp((-t**2)/alpha**2)*np.exp(1j*2*np.pi*t)


def averaging_signal_linear(signal):
    k = int(0.01*len(signal))
    new_signal = [1/(2*k+1)*sum(signal[i:i+k]) for i in range(len(signal))]
    return new_signal

def averaging_signal_quad(signal, time):
    w = find_half_max(signal)
    g = np.exp([-4*cmath.log(2, np.e)*t/w**2 for t in time])

    k = int(0.01*len(signal))
    new_signal = [sum(signal[i:i + k]*g[i:i+k]) for i in range(len(signal))]
    return new_signal

def median_filter(signal, window_size):
    filtered_signal = []
    padding = (window_size - 1) // 2
    for i in range(padding, len(signal) - padding):
        window = signal[i - padding:i + padding + 1]
        sorted_window = sorted(window)
        median = sorted_window[len(sorted_window) // 2]
        filtered_signal.append(median)
    return filtered_signal

def spectrum_interpolation(signal, time):
    signal[time[0]:time[1]] = np.zeros(time[1]-time[0])

    window_duration = time[1]-time[0]

    window_1 = signal[time[0]-window_duration:time[0]]
    window_2 = signal[time[1]:time[1]+window_duration]

    window_1_spect = np.fft.fft(window_1)
    window_2_spect = np.fft.fft(window_2)

    window_middle = (window_1_spect+window_2_spect)/2
    middle_signal = np.real(np.fft.ifft(window_middle))
    signal[time[0]:time[1]] = middle_signal

    return signal

def discrease_sampling(signal, freqs, n=2, sum_signal=True):

    new_nyiquist = len(signal)/2*n
    coefficients = butterworth_filter_low(freqs, new_nyiquist)
    if sum_signal:
        filtered_signal = (sum([np.abs(imag) * real for imag, real in zip(coefficients, signal)]))
    else:
        filtered_signal = [np.abs(imag) * real for imag, real in zip(coefficients, signal)]
    return filtered_signal

def quadro_method(K,f,a,b,h):
    x=np.arange (a, b, h)
    x=x.reshape(len(x),1)
    n=len(x)
    wt=1/2
    wj=1
    A=np.zeros((n, n))
    for i in range(n):
        A[i][0]=-h*wt*K(x[i],x[0])
        for j in range(1,n-1,1):
            A[i][j]= -h*wj*K(x[i],x[j])
        A[i][n-1]= -h*wt*K(x[i],x[n-1])
        A[i][i]= A[i][i]+ 1
    B = np.zeros((n,1))
    for j in range(n):
        B[j][0] = f(x[j])
    y=np.linalg.solve(A, B)
    return y


alpha = lambda t: [-t,t, t**2, t**3, t**4]
beta= lambda t: [1 , 1, t, 0.5*t**2, 1/6*t**3]

def bfun(t,m,f):
    return beta(t)[m]*f(t)

def Aijfun(t,m,k):
    return beta(t)[m]*alpha(t)[k]

def kernel_approx(a, b, f,t,Lambda):
    m=len(alpha(0)) # определяем размер alpha
    M=np.zeros((m,m))
    r=np.zeros((m,1))

    for i in range(m):
        r[i]=integrate.quad(bfun, a, b,args=(i,f))[0]
        for j in range(m):
            M[i][j]=-Lambda*integrate.quad(Aijfun, a, b,args=(i,j))[0]


    for i in range(m):
        M[i][i] =M[i][i]+1


    c=np.linalg.solve(M, r)
    aij = np.array(alpha(t))

    return Lambda*(np.sum(c[:,np.newaxis]*aij,axis=0))+f(t)

def galkin_petrov(b, a, psi, K, f, lam, phi):
    for i in range(2):
        b[i] = lam * quad(lambda x: psi[i](x) * quad(lambda s: K(x, s) * f(s), -1.001, 1.001)[0], -1.001, 1.001)[0]
        for j in range(2):
            a[i][j] = quad(lambda x: phi[i](x) * psi[j](x), -1.001, 1.001)[0] - lam * \
                      quad(lambda x: psi[i](x) * quad(lambda s: K(x, s) * phi[j](s), -1.001, 1.001)[0], -1.001, 1.001)[
                          0]

    return a, b



