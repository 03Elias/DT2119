# DT2119, Lab 1 Feature Extraction
import numpy as np
import matplotlib.pyplot as plt
from lab1_tools import lifter, trfbank
from scipy.signal import windows

# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    

    samples = np.asarray(samples)
    signal_length = samples.shape[0]

    if winlen <= 0 or winshift <= 0:
        raise ValueError("winlen and winshift must be positive ints")

    if signal_length < winlen:
        return np.empty((0, winlen), dtype=samples.dtype)

    n_frames = 1 + (signal_length - winlen) // winshift
    starts = np.arange(n_frames) * winshift
    indices = starts[:, None] + np.arange(winlen)[None, :]
    return samples[indices]
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    input = np.asarray(input)
    win = windows.hamming(input.shape[1], sym=False)
    return input * win
def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    input = np.asarray(input)
    nfft = input.shape[1]
    fbank = trfbank(samplingrate, nfft)
    mel_energies = np.dot(input, fbank.T)
    return np.log(np.maximum(mel_energies, np.finfo(float).eps))

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    LD = np.zeros((len(x), len(y)), dtype=float)
    for i in range(len(x)):
        for j in range(len(y)):
            LD[i, j] = dist(x[i], y[j])

    if LD.ndim != 2 or LD.shape[0] == 0 or LD.shape[1] == 0:
        raise ValueError("Local-distance matrix must be 2D and non-empty.")

    n, m = LD.shape
    AD = np.full((n, m), np.inf, dtype=float)
    AD[0, 0] = LD[0, 0]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            AD[i, j] = LD[i, j] + min(
                AD[i - 1, j] if i > 0 else np.inf,
                AD[i, j - 1] if j > 0 else np.inf,
                AD[i - 1, j - 1] if (i > 0 and j > 0) else np.inf,
            )

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        candidates = []
        if i > 0 and j > 0:
            candidates.append((AD[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((AD[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((AD[i, j - 1], i, j - 1))
        _, i, j = min(candidates, key=lambda t: t[0])
        path.append((i, j))
    path.reverse()

    d = AD[-1, -1] / (n + m)
    return d, LD, AD, path



