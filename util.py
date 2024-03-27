import numpy as np
import math
import argparse
import torch
import scipy.linalg
import random
from kornia import morphology, filters

SAMPLING_RATE = None


def set_sampling_rate(sr):
    global SAMPLING_RATE
    SAMPLING_RATE = sr


def clear_dir(p):
    """
    Delete the contents of the directory at p
    """
    if not p.is_dir():
        return
    for f in p.iterdir():
        if f.is_file():
            f.unlink()
        else:
            clear_dir(f)
            f.rmdir()


def format_time(seconds):
    """

    :param seconds:
    :return: a dictionary with the keys 'h', 'm', 's', that is the amount of hours, minutes, seconds equal to 'seconds'
    """
    hms = [seconds // 3600, (seconds // 60) % 60, seconds % 60]
    hms = [int(t) for t in hms]
    labels = ['h', 'm', 's']
    return {labels[i]: hms[i] for i in range(len(hms))}


def time_string(seconds):
    """
    Returns a string with the format "0h 0m 0s" that represents the amount of time provided
    :param seconds:
    :return: string
    """
    t = format_time(seconds)
    return "{}h {}m {}s".format(t['h'], t['m'], t['s'])


def rms(audio, sr):
    return np.sqrt(np.mean(audio**2))


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def spectrum(audio, sr):
    """
    Return the magnitude spectrum and the frequency bins

    :returns amps, freqs: where the ith element of amps is the amplitude of the ith frequency in freqs
    """
    # TODO: Should this be the normalized magnitude spectrum?
    # calculate amplitude spectrum
    N = next_power_of_2(audio.size)
    fft = np.fft.rfft(audio, N)
    amplitudes = abs(fft)

    # get frequency bins
    frequencies = np.fft.rfftfreq(N, d=1. / sr)

    return amplitudes, frequencies

def centroid(audio, sr):
    """
    Compute the spectral centroid of the given audio.
    Spectral centroid is the weighted average of the frequencies, where each frequency is weighted by its amplitude

    the centroid is the sum across each frequency f and amplitude a: f * a / sum(a)

    :param audio: audio as a numpy array
    :param sr: the sampling rate of the audio
    """

    amps, freqs = spectrum(audio, sr)

    if sum(amps) == 0:
        return 0

    return sum(amps * freqs) / sum(amps)


def spread(audio, sr):
    """
    Compute the spectral spread of the given audio
    Spectral spread is the average each frequency component weighted by its amplitude and subtracted by the spectral centroid

    spread = sqrt(sum(amp(k) * (freq(k) - centroid)^2) / sum(amp))
    """
    amps, freqs = spectrum(audio, sr)
    cent = centroid(audio, sr)

    if sum(amps) == 0:
        return 0

    return math.sqrt(sum(amps * (freqs - cent)**2) / sum(amps))


def skewness(audio, sr):
    """
    Compute the spectral skewness

    Skewness is the sum of (freq - centroid)^3 * amps divided by (spread^3 times the sum of the amps)

    """
    amps, freqs = spectrum(audio, sr)

    if sum(amps) == 0:
        return 0

    cent = centroid(audio, sr)
    spr = spread(audio, sr)

    return sum(amps * (freqs - cent)**3) / (spr**3 * sum(amps))


def kurtosis(audio, sr):
    """
    Compute the spectral kurtosis

    Kurtosis is the sum of (freq - centroid)^4 * amp divided by (the spread^4 times the sum of the amps)

    """
    amps, freqs = spectrum(audio, sr)

    if sum(amps) == 0:
        return 0

    cent = centroid(audio, sr)
    spr = spread(audio, sr)

    return sum(amps * (freqs - cent)**4) / (spr**4 * sum(amps))


def moments(audio, sr):
    """
    Return the four statistical moments that make up the spectral shape: centroid, spread, skewness, kurtosis
    """
    amps, freqs = spectrum(audio, sr)

    if sum(amps) == 0:
        return 0, 0, 0, 0

    cent = sum(amps * freqs) / sum(amps)
    spr = math.sqrt(sum(amps * (freqs - cent)**2) / sum(amps))
    skew = sum(amps * (freqs - cent)**3) / (spr**3 * sum(amps))
    kurt = sum(amps * (freqs - cent)**4) / (spr**4 * sum(amps))

    return cent, spr, skew, kurt


def flux(spectrum1, spectrum2):
    """
    Sum of absolute value of current amp minus prev frame's amp
    @param spectrum1: an np array of amplitudes, which is spectrum(audio, sr)[0]
    @param spectrum2: an np array of amplitudes, which is spectrum(audio, sr)[0]
    @return: spectral flux
    """
    if spectrum2 is None or spectrum1 is None:
        return 0
    spectral_flux = 0
    assert spectrum1.size == spectrum2.size
    for i in range(spectrum1.size):
        diff = abs(spectrum1[i] - spectrum2[i])
        spectral_flux += diff
    return spectral_flux


def add_scalar(x, a):
    """

    @param x:
    @param a:
    @return:
    """
    return x + (torch.ones_like(x) * a)


def add_rms(x, audio):
    """
    Add the rms of audio to the latent tensor x
    @param x: latent tensor
    @param audio:
    @return: same shape as x
    """
    return x + (torch.ones_like(x) * rms(audio))


def add_gaussian_rms(x, audio):
    """
    Add a matrix made of random samples from a normal gaussian
    @param x: latent tensor
    @param audio:
    @return:
    """
    return x + torch.normal(torch.zeros_like(x), torch.ones_like(x)) * rms(audio)


def add_centroid(x, audio):
    """
    Add a tensor full of the spectral centroid to x
    @param x:
    @param audio:
    @return:
    """
    scale = 1./1000
    return x + (torch.ones_like(x) * centroid(audio, SAMPLING_RATE) * scale)


def add_spread(x, audio):
    """
    Add a tensor full of the spectral spread to x
    @param x:
    @param audio:
    @return:
    """
    scale = 1./10
    return x + (torch.ones_like(x) * spread(audio, SAMPLING_RATE) * scale)


def add_skewness(x, audio):
    """
    Add a tensor full of the spectral skewness to x
    @param x:
    @param audio:
    @return:
    """
    scale = 1./10
    return x + (torch.ones_like(x) * skewness(audio, SAMPLING_RATE) * scale)


def add_kurtosis(x, audio):
    """
    Add a tensor full of the spectral kurtosis to x
    @param x:
    @param audio:
    @return:
    """
    scale = 1./10
    return x + (torch.ones_like(x) * kurtosis(audio, SAMPLING_RATE) * scale)


def add_full(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    added to every element
    """
    return lambda x: x + (torch.ones_like(x) * r)


def add_sparse(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    added to 25% of the elements
    """
    return lambda x: x + ((torch.rand_like(x) < 0.05) * r)


def add_noise(r):
    """
    Return a fn that adds Gaussian noise mutliplied by r to x
    """

    return lambda x: x + (torch.randn_like(x) * r)



def subtract_full(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    subtracted from every element
    """
    return lambda x: x - (torch.ones_like(x) * r)


def threshold(r):
    def thresh(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: y if abs(y) >= r else 0)
        x = x.to(device)
        return x
    return thresh


def soft_threshold(r):
    """
    Return a fn that applies soft thresholding to x
    In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    source: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    """
    def fn(x):
        x = x / x.abs() * torch.maximum(x.abs() - r, torch.zeros_like(x))
        return x
    return fn


def soft_threshold2(r):
    """
    Return a fn that applies soft thresholding to x
    In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    source: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    """
    def fn(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: 0 if abs(y) < r else y*(1-r))
        x = x.to(device)
        return x
    return fn


def inversion(r):
    def foo(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: 1./r - y)
        x = x.to(device)
        return x
    return foo


def inversion2():
    return lambda x: -1 * x


def log(r):
    return lambda x: torch.log(x)


def power(r):
    return lambda x: torch.pow(x, r)

def add_dim(r, dim, i):
    def foo(x):
        """
        Add value r to the given dim at index i
        dim = 0 means adding in "z" dimension (shape = 4)
        dim = 1 means adding to row i
        dim = 2 means adding to column i
        @return: modified x
        """
        if dim == 0:
            for a in range(x.shape[0]):
                x[a, i, i] += r
        elif dim == 1:
            x[:, i:i + 1] += r
        elif dim == 2:
            x[:, :, i:i + 1] += r
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo

def add_rand_cols(r, k):
    """
    Return a fn that will add value 'r' to a fraction of the cols of a tensor
    Assume x is a 3-tensor and rows refer to the third dimension
    k should be between 0 and 1
    """
    def foo(x):
        dim = x.shape[1]
        cols = random.sample(range(dim), int(k * dim))
        for col in cols:
            x[:, :, col:col + 1] += r
        return x

    return foo


def add_rand_rows(r, k):
    """
    Return a fn that will add value 'r' to a fraction of the rows of a tensor
    Assume x is a 3-tensor and rows refer to the second dimension
    k should be between 0 and 1
    """
    def foo(x):
        dim = x.shape[2]
        rows = random.sample(range(dim), int(k * dim))
        for row in rows:
            x[:, row:row + 1] += r
        return x

    return foo


def invert_dim(r, dim, i):
    def foo(x):
        """
        Apply inversion (1/r. - x) at the given dim at index i
        dim = 0 means applying in "z" dimension (shape = 4)
        dim = 1 means applying to row i
        dim = 2 means applying to column i
        @return: modified x
        """
        invert = inversion(r)
        if dim == 0:
            for a in range(x.shape[0]):
                x[a, i, i] = invert(x[a, i, i])
        elif dim == 1:
            x[:, i:i + 1] = invert(x[:, i:i + 1])
        elif dim == 2:
            x[:, :, i:i + 1] += invert(x[:, :, i:i + 1])
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo


def apply_to_dim(func, r, dim, i):
    def foo(x):
        """
        Apply func at the given dim at i
        dim = 0 means applying in "z" dimension (shape = 4)
        dim = 1 means applying to row i
        dim = 2 means applying to column i
        @return: modified x
        """
        fn = func(r)
        if dim == 0:
            for a in range(x.shape[0]):
                try:
                    x[a, i[0], i[1]] = fn(x[a, i[0], i[1]])
                except TypeError:
                    x[a, i, i] = fn(x[a, i, i])
        elif dim == 1:
            try:
                x[:, i[0]:i[1]] = fn(x[:, i[0]:i[1]])
            except TypeError:
                x[:, i:i + 1] = fn(x[:, i:i + 1])
        elif dim == 2:
            try:
                x[:, :, i[0]:i[1]] = fn(x[:, :, i[0]:i[1]])
            except TypeError:
                x[:, :, i:i + 1] = fn(x[:, :, i:i + 1])
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo


def apply_sparse(func, sparsity):
    """
    return a function that applies the given function a random fraction of the elements, as determined by 'sparsity'
    0 < sparsity < 1
    """
    def fn(x):
        mask = torch.rand_like(x) < sparsity
        x = (x * ~mask) + (func(x) * mask)
        return x
    return fn


def add_normal(r):
    """
    Add a 2D normal gaussian (bell curve) to the center of the tensor
    """
    def foo(x):
        # chatgpt wrote this
        # Define the size of the matrix
        size = 64

        # Generate grid coordinates centered at (0,0)
        a = np.linspace(-5, 5, size)
        b = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(a, b)

        # Standard deviation
        sigma = 1.5  # You can adjust this value as desired

        # Generate a 2D Gaussian distribution with peak at the center and specified standard deviation
        Z = np.exp(-0.5 * ((X / sigma) ** 2 + (Y / sigma) ** 2)) / (2 * np.pi * sigma ** 2)
        Z *= r
        Z = torch.from_numpy(Z).to(x.get_device())

        for i in range(x.shape[0]):
            x[i] += Z
        return x
    return foo


def tensor_exp(r):
    """
    Return a fn that computes the matrix exponential of a given tensor
    """
    def foo(x):
        device = x.get_device()
        x = x.cpu().numpy()
        x = scipy.linalg.expm(x)
        x = torch.from_numpy(x).to(device)
        return x
    return foo


def rotate_z(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "z" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c, -1 * s, 0, 0],
            [s,    c,   0, 0],
            [0,    0,   1, 0],
            [0,    0,   0, 1]
        ]
        op = torch.tensor(rotation_matrix).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def rotate_x(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "x" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [1, 0,   0,    0],
            [0, c, -1 * s, 0],
            [0, s,   c,    0],
            [0, 0,   0,    1]
        ]
        op = torch.tensor(rotation_matrix).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def rotate_y(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "y" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c,      0, s, 0],
            [0,      1, 0, 0],
            [-1 * s, 0, c, 0],
            [0,      0, 0, 1]
        ]
        op = torch.tensor(rotation_matrix).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def rotate_y2(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "y" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c,      0, 0, s],
            [0,      1, 0, 0],
            [0,      0, 1, 0],
            [-1 * s, 0, 0, c]
        ]
        op = torch.tensor(rotation_matrix).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def reflect(r):
    """
    Return a fn that reflects across the given dimension r
    r can be 0, 1, 2, or 3
    """
    def fn(x):
        device = x.get_device()
        op = torch.eye(4)  # identity matrix
        op[r, r] *= -1
        op = op.to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def hadamard1(r):
    def fn(x):
        device = x.get_device()
        h = scipy.linalg.hadamard(4)
        op = torch.tensor(h).to(torch.float32).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def hadamard2(r):
    def fn(x):
        device = x.get_device()
        h = scipy.linalg.hadamard(64)
        op = torch.tensor(h).to(torch.float32).to(device)
        x = torch.tensordot(x, op, dims=[[1], [1]])
        return x
    return fn


def apply_both(fn1, fn2, r):
    def fn(x):
        return fn1(fn2(r))
    return fn



def normalize(func):
    """
    First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)  # first apply the network bending function
        # then normalize the result
        max = x.abs().max()
        x = x / max
        return x
    return fn


def normalize2(func):
    """
    First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)  # first apply the network bending function
        # then normalize the result
        x = torch.nn.functional.normalize(x, dim=0)
        return x
    return fn


def normalize3(func):
    """
        First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)
        x = x - x.mean()
        return x
    return fn


def normalize4(func, dim=0):
    """
        First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)
        x = x - torch.mean(x, dim=dim, keepdim=True)
        return x
    return fn


def gradient():
    def fn(x):
        x = x.unsqueeze(0)
        kernel = torch.ones(4, 4).to(x.get_device())
        x = morphology.gradient(x, kernel)
        x = x.squeeze(0)
        return x
    return fn


def dilation(r):
    def fn(x):
        x = x.unsqueeze(0)
        kernel = torch.ones(r, r).to(x.get_device())
        x = morphology.dilation(x, kernel)
        x = x.squeeze(0)
        return x
    return fn


def erosion(r):
    def fn(x):
        x = x.unsqueeze(0)
        kernel = torch.ones(r, r).to(x.get_device())
        x = morphology.erosion(x, kernel)
        x = x.squeeze(0)
        return x
    return fn


def sobel(r=True):
    def fn(x):
        x = x.unsqueeze(0)
        x = filters.sobel(x, normalized=r)
        x = x.squeeze(0)
        return x
    return fn



def absolute(r):
    """
    Return a fn that computes the absolute value of a tensor
    """
    def fn(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: abs(y))
        x = x.to(device)
        return x
    return fn


def log(r=math.e):
    """
    Return a fn that computes the log of a tensor with base r. Must first ensure that it is non-negative
    """
    def fn(x):
        device = x.get_device() if x.get_device() is not None else 'cpu'
        x = x.cpu()
        x = x.apply_(lambda y: abs(y))
        x = x.to(device)
        return torch.log(x) / math.log(r)
    return fn

def clamp(r):
    """
    Return a fn that clamps a tensor between min and max
    """
    def fn(x):
        min, max = r
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: min if y < min else max if y > max else y)
        x = x.to(device)
        return x
    return fn

def scale(r):
    """
    Return a fn that scales a tensor between min and max based on the tensor's min and max
    """
    def fn(x):
        min, max = r
        device = x.get_device()
        x = x.cpu()
        xmin = x.min()
        xmax = x.max()  
        x = x.apply_(lambda y: (y - xmin) / (xmax - xmin) *(max - min) + min)
        x = x.to(device)
        return x
    return fn


def run_argparse():
    """
    Run the command line argument parse

    Returns: the object returned from parser.parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=str,
        help="path to input audio file",
        required=True
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="frames per second",
        required=True
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="for img2img: strength for noising/unnoising. "
             "1.0 corresponds to full destruction of information in init image"
    )
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )
    # parser.add_argument(
    #     "--outdir",
    #     type=str,
    #     nargs="?",
    #     help="dir to write results to",
    #     default="outputs/audio2img-samples"
    # )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        default=True,
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    args = parser.parse_args()
    return args


