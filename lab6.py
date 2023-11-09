import numpy as np
from dsp import signal_generator, morle_wavelet, draw_plot, mexico_hat_wavelet, haare_wavelet, welve
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import librosa
import librosa.display

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
    alpha = 3
    discrete_freq = 300
    frequencies = [25, 50, 75]
    freq_wavelets = 10

    time = np.linspace(0, 1, discrete_freq)
    time_wavelet = np.linspace(-2, 2, freq_wavelets)
    omega = np.linspace(-10, 10, discrete_freq)

    signal = signal_generator(frequencies, time, np.cos)

    morle, _ = morle_wavelet(omega, time_wavelet, alpha)
    spectro_gram = np.convolve(signal, morle, mode='same')

    plt.specgram(spectro_gram, Fs=discrete_freq, cmap="inferno", scale_by_freq=True, mode='magnitude', detrend='linear',
                 sides='onesided')
    plt.show()

def task_7():

    braindat = sio.loadmat(r'data/Lab6_Data.mat')
    time_vec = braindat['timevec'][0]
    s_rate = braindat['srate'][0]
    data = braindat['data'][0]

    discrete_freq = 50

    signal = np.linspace(8, 70, discrete_freq)
    time = np.arange(-2, 2, 1 / s_rate)

    y = ([welve(t) for t in time_vec])
    y = np.convolve(data, y)

    lst_signal = []

    for cnt in range(discrete_freq):
        lst_signal.append(np.exp(1j * 2 * np.pi * signal[cnt] * time) * np.exp(-(4 * np.log(2) * time ** 2) / 0.2 ** 2))

    dataX = scipy.fftpack.fft(y, len(time_vec) + len(time) - 1)

    tf = []

    for sign in lst_signal:
        waveX = scipy.fftpack.fft(sign, len(time_vec) + len(time) - 1)
        waveX = waveX / np.max(waveX)

        conv_res = scipy.fftpack.ifft(waveX * dataX)
        conv_res = conv_res[len(time) // 2 - 1: -len(time) // 2]
        tf.append(np.abs(conv_res) ** 2)

    plt.pcolormesh(time_vec, signal, tf, vmin=0, vmax=1e3, cmap='gist_heat')
    plt.show()

def task_6():
    y, sr = librosa.load('data/Rick_Astley_-_Never_Gonna_Give_You_Up_47958276.mp3')

    spec = librosa.feature.melspectrogram(y=y, sr=sr)

    spec_db = librosa.power_to_db(spec, ref=np.max)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    img = librosa.display.specshow(spec_db, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogram')
    plt.show()


task_6()


