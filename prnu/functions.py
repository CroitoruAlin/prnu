# -*- coding: UTF-8 -*-
"""
@author: Mirco Ceccarelli (mirco.ceccarelli@stud.unifi.it)
@author: Francesco Argentieri (francesco.argentieri@stud.unifi.it)
Università degli Studi di Firenze 2021
"""

from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, cpu_count
import pywt
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters
from tqdm import tqdm

import os.path
import cv2
import numpy as np
import torch
from einops import rearrange
from utils import utils_image as util
from utils import utils_model
import time
class ArgumentError(Exception):
    pass


"""
Extraction functions
"""

def noise_extract_restomer(im: np.ndarray, model=None, levels: int = 100, sigma: int = 100) -> np.ndarray:

    im_noise = im+ np.random.normal(0, sigma/255., im.shape)
    im_noise = rearrange(im_noise, "b h w c-> b c h w").to("cuda").float()
    with torch.no_grad():
        restored = model(im_noise).cpu()

    restored = torch.clamp(restored,0,1)
    restored = rearrange(restored, "b c h w-> b h w c")
    noise = (im-restored)*255
    return noise.numpy()



# Performs the noise extraction operation via the DRUNET network.
def noise_extract_drunet(im: np.ndarray, model=None, levels: int = 100, sigma: int = 100) -> np.ndarray:
    """
        Extract noise residual from a single image
        :param im: grayscale or color image, np.uint8
        :param levels: number of noise levels (try: 15, 50, 100)
        :param sigma: estimated noise power (try: 15, 50, 100)
        :return: noise residual
    """

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    x8 = False
    noise_level_img = levels        # set AWGN noise level for noisy image
    noise_level_model = sigma       # set noise level for model
    device = next(model.parameters()).device
    # ------------------------------------
    # (1) img_L (Low-quality images)
    # ------------------------------------
    img_H = im
    # print(img_H.shape)
    img_L = util.uint2single(img_H)

    # Add noise without clipping
    np.random.seed(seed=0)  # for reproducibility
    img_L += np.random.normal(0, noise_level_img / 255., img_L.shape)

    img_L = util.single2tensor4(img_L)
    img_L = torch.cat(
        (img_L, torch.FloatTensor([noise_level_img / 255.]).repeat(1 if len(img_L.shape)==3 else img_L.shape[0], 1, img_L.shape[2], img_L.shape[3])), dim=1)
    img_L = img_L.to(device)

    # ------------------------------------
    # (2) img_E (Estimated images)
    # ------------------------------------

    if not x8 and img_L.size(2) // 8 == 0 and img_L.size(3) // 8 == 0:
        img_E = model(img_L)
    elif not x8 and (img_L.size(2) // 8 != 0 or img_L.size(3) // 8 != 0):
        img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
    elif x8:
        img_E = utils_model.test_mode(model, img_L, mode=3)

    img_E = util.tensor2uint(img_E)
    noise = img_H - img_E

    return noise

def model_creation():
    model_name = 'drunet_color'     # set denoiser model, 'drunet_gray' or 'drunet_color'
    x8 = False                      # default: False, x8 to boost performance

    if 'color' in model_name:
        n_channels = 3              # 3 for color image
    else:
        n_channels = 1              # 1 for grayscale image

    model_pool = 'model_zoo'        # fixed

    model_path = os.path.join(model_pool, model_name + '.pth')
    device = torch.device('cuda')
    torch.cuda.empty_cache()

        # ----------------------------------------
        # Load model
        # ----------------------------------------

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model,x8,device


def extract_single(im: np.ndarray,
                   model=None,
                   levels: int = 50,
                   sigma: int = 50,
                   wdft_sigma: float = 0) -> np.ndarray:
    """
    Extract noise residual from a single image
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels (try: 15, 50, 100)
    :param sigma: estimated noise power (try: 15, 50, 100)
    :param wdft_sigma: estimated DFT noise power
    :return: noise residual
    """

    # To use the Polimi noise extract function.
    #W = noise_extract(im, levels, sigma)

    # To use the DRUNet noise extract function.
    W = noise_extract_restomer(im, model, levels, sigma)
    result = np.zeros(W.shape[:3])
    if len(W.shape)>3:
        for i in range(W.shape[0]):
            result[i] = rgb2gray(W[i]/255.)
            result[i] = zero_mean_total(result[i])
            W_std = result[i].std(ddof=1) if wdft_sigma == 0 else wdft_sigma
            result[i] = wiener_dft(result[i], W_std).astype(np.float32)
        W=result
    else:
        W = rgb2gray(W)
        W = zero_mean_total(W)
        W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
        W = wiener_dft(W, W_std).astype(np.float32)

    return W

def extract_single_drunet(im: np.ndarray,
                   model=None,
                   levels: int = 100,
                   sigma: int = 100,
                   wdft_sigma: float = 0) -> np.ndarray:
    """
    Extract noise residual from a single image
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels (try: 15, 50, 100)
    :param sigma: estimated noise power (try: 15, 50, 100)
    :param wdft_sigma: estimated DFT noise power
    :return: noise residual
    """

    # To use the Polimi noise extract function.
    #W = noise_extract(im, levels, sigma)

    # To use the DRUNet noise extract function.
    start_time = time.time()
    W = noise_extract_drunet(im, model, levels, sigma)
    end_time = time.time()
    print("noise extraction", end_time-start_time)
    result = np.zeros(W.shape[:3])
    if len(W.shape)>3:
        W = rgb2gray_batched(W)
        W = zero_mean_total_batched(W)
        W_std = W.std(ddof=1, axis=(1, 2))
        W = wiener_dft_batched(W, W_std).astype(np.float32)
    else:
        W = rgb2gray(W)
        W = zero_mean_total(W)
        W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
        W = wiener_dft(W, W_std).astype(np.float32)

    return W


# Performs the noise extraction operation through a model-based algorithm.
def noise_extract(im: np.ndarray, levels: int = 4, sigma: float = 5) -> np.ndarray:
    """
    NoiseExtract as from Binghamton toolbox.
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: noise residual
    """

    assert (im.dtype == np.uint8)
    assert (im.ndim in [2, 3])

    im = im.astype(np.float32)
    noise_var = sigma ** 2

    if im.ndim == 2:
        im.shape += (1,)

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

        wlet = None
        while wlet is None and levels > 0:
            try:
                wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
            except ValueError:
                levels -= 1
                wlet = None
        if wlet is None:
            raise ValueError('Impossible to compute Wavelet filtering for input size: {}'.format(im.shape))

        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = wiener_adaptive(wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, 'db4')
        try:
            W[:, :, ch] = wrec
        except ValueError:
            W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
            W[:, :, ch] = wrec

    if W.shape[2] == 1:
        W.shape = W.shape[:2]

    W = W[:im.shape[0], :im.shape[1]]
    return W


def noise_extract_compact(args):
    """
    Extract residual, multiplied by the image. Useful to save memory in multiprocessing operations
    :param args: (im, levels, sigma), see noise_extract for usage
    :return: residual, multiplied by the image
    """

    # To use the Polimi noise extract function.
    #w = noise_extract(*args)

    # To use the DRUNet noise extract function.
    w = noise_extract_restomer(*args)
    im = args[0]
    try:
        return w.float()
    except:
        return w.astype(np.float32)

def noise_extract_compact_drunet(args):
    """
    Extract residual, multiplied by the image. Useful to save memory in multiprocessing operations
    :param args: (im, levels, sigma), see noise_extract for usage
    :return: residual, multiplied by the image
    """

    # To use the Polimi noise extract function.
    #w = noise_extract(*args)

    # To use the DRUNet noise extract function.
    w = noise_extract_drunet(*args)
    im = args[0]
    try:
        return w.float()
    except:
        return w.astype(np.float32)


def extract_multiple_aligned(imgs: list, levels: int = 100, sigma: int = 100, processes: int = None,
                             batch_size=cpu_count(), tqdm_str: str = '') -> np.ndarray:
    """
    Extract PRNU from a list of images. Images are supposed to be the same size and properly oriented
    :param tqdm_str: tqdm description (see tqdm documentation)
    :param batch_size: number of parallel processed images
    :param processes: number of parallel processes
    :param imgs: list of images of size (H,W,Ch) and type np.uint8
    :param levels: number of wavelet decomposition levels (try: 15, 50, 100)
    :param sigma: estimated noise power (try: 15, 50, 100)
    :return: PRNU
    """
    assert (isinstance(imgs[0], np.ndarray))
    assert (imgs[0].ndim == 3)
    assert (imgs[0].dtype == np.uint8)

    h, w, ch = imgs[0].shape

    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)

    if processes is None or processes > 1:
        args_list = []
        for im in imgs:
            args_list += [(im, levels, sigma)]
        pool = Pool(processes=processes)

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (1/2)'), dynamic_ncols=True):
            nni = pool.map(inten_sat_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for ni in nni:
                NN += ni
            del nni

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (2/2)'), dynamic_ncols=True):
            wi_list = pool.map(noise_extract_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for wi in wi_list:
                RPsum += wi
            del wi_list

        pool.close()

    else:  # Single process
        for im in tqdm(imgs, disable=tqdm_str is None, desc=tqdm_str, dynamic_ncols=True):
            RPsum += noise_extract_compact((im, None, levels, sigma))
            NN += (inten_scale(im) * saturation(im)) ** 2

    K = RPsum / (NN + 1)
    K = rgb2gray(K)
    K = zero_mean_total(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)

    return K


def cut_ctr(array: np.ndarray, sizes: tuple) -> np.ndarray:
    """
    Cut a multi-dimensional array at its center, according to sizes
    :param array: multidimensional array
    :param sizes: tuple of the same length as array.ndim
    :return: multidimensional array, center cut
    """
    array = array.copy()
    if not (array.ndim == len(sizes)):
        raise ArgumentError('array.ndim must be equal to len(sizes)')
    for axis in range(array.ndim):
        axis_target_size = sizes[axis]
        axis_original_size = array.shape[axis]
        if axis_target_size > axis_original_size:
            raise ValueError(
                'Can\'t have target size {} for axis {} with original size {}'.format(axis_target_size, axis,
                                                                                      axis_original_size))
        elif axis_target_size < axis_original_size:
            axis_start_idx = (axis_original_size - axis_target_size) // 2
            axis_end_idx = axis_start_idx + axis_target_size
            array = np.take(array, np.arange(axis_start_idx, axis_end_idx), axis)
    return array



def rgb2gray_batched(im: np.ndarray) -> np.ndarray:
    """
    Efficient RGB → Grayscale conversion (Binghamton weights).
    Works with single image (H,W,3) or batch (B,H,W,3).
    Returns float32 grayscale with shape (H,W) or (B,H,W).
    """
    weights = np.array([0.29893602, 0.58704307, 0.11402090], dtype=np.float32)

    if im.ndim == 2:
        # already grayscale
        return im.astype(np.float32)

    if im.ndim == 3:
        if im.shape[2] == 1:
            return im[..., 0].astype(np.float32)
        elif im.shape[2] == 3:
            return np.tensordot(im, weights, axes=([-1],[0])).astype(np.float32)
        else:
            raise ValueError("Input image must have 1 or 3 channels")

    if im.ndim == 4:
        if im.shape[-1] == 1:
            return im[..., 0].astype(np.float32)
        elif im.shape[-1] == 3:
            return np.tensordot(im, weights, axes=([-1],[0])).astype(np.float32)
        else:
            raise ValueError("Input images must have 1 or 3 channels")

    raise ValueError("Input must be (H,W), (H,W,1/3), or (B,H,W,1/3)")

def zero_mean_total_batched(im: np.ndarray) -> np.ndarray:
    """
    ZeroMeanTotal as from Binghamton toolbox, vectorized.
    Works on single images (H,W) or batches (B,H,W).
    Applies zero-mean separately to the 4 checkerboard subgrids.
    Returns float32 array with same shape.
    """
    im = im.astype(np.float32, copy=False)

    if im.ndim == 2:   # (H,W)
        out = im.copy()
        for i in (0,1):
            for j in (0,1):
                sub = out[i::2, j::2]
                out[i::2, j::2] = sub - sub.mean(dtype=np.float32)
        return out

    elif im.ndim == 3:  # (B,H,W)
        out = im.copy()
        for i in (0,1):
            for j in (0,1):
                sub = out[:, i::2, j::2]  # shape (B,h,w)
                means = sub.mean(axis=(1,2), keepdims=True, dtype=np.float32)
                out[:, i::2, j::2] = sub - means
        return out

    else:
        raise ValueError("Input must be (H,W) or (B,H,W)")


def wiener_dft_batched(im, sigma) -> np.ndarray:
    """
    Adaptive Wiener filter applied to the 2D FFT of the image.
    Works with (H,W) or (B,H,W). Supports scalar sigma or per-image sigma (B,).

    Parameters
    ----------
    im : np.ndarray
        Input image(s), shape (H,W) or (B,H,W). Real-valued.
    sigma : float or np.ndarray
        Estimated noise std. If batched input, can be scalar or shape (B,).

    Returns
    -------
    np.ndarray
        Filtered image(s), same shape as `im`, dtype float32.
    """
    im = np.asarray(im)
    if im.ndim == 2:
        im_in = im[np.newaxis, ...]        # -> (1,H,W)
        batched = False
    elif im.ndim == 3:
        im_in = im                          # (B,H,W)
        batched = True
    else:
        raise ValueError("`im` must be (H,W) or (B,H,W)")

    B, H, W = (im_in.shape[0], im_in.shape[1], im_in.shape[2])

    # Prepare sigma (noise std) per image
    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.ndim == 0:
        sigma = np.full((B,), float(sigma), dtype=np.float32)
    elif sigma.shape != (B,):
        raise ValueError(f"`sigma` must be scalar or shape ({B},), got {sigma.shape}")
    noise_var = sigma ** 2
    # reshape for broadcasting across (H,W)
    noise_var_ = noise_var[:, None, None]   # (B,1,1)

    # FFT2 over last two axes (vectorized over batch)
    im_fft = np.fft.fft2(im_in, axes=(-2, -1))                   # (B,H,W), complex
    # Magnitude normalized by sqrt(H*W), same as original
    denom = (H * W) ** 0.5
    mag = np.abs(im_fft) / denom                                  # (B,H,W), real

    # --- Apply adaptive Wiener on magnitude ---
    # Expecting a function `wiener_adaptive(x, noise_var)` that operates per-image.
    # If your `wiener_adaptive` supports batched input, call once; otherwise loop.
    # Below we implement a simple per-pixel Wiener gain on the magnitude domain:
    #
    #   S = max(mag^2 - noise_var, 0)   (signal power estimate)
    #   G = S / (S + noise_var)         (Wiener gain in [0,1])
    #
    # This mirrors the common Wiener filter form without requiring an external function.
    #
    power = mag**2                                               # (B,H,W)
    signal_power = np.maximum(power - noise_var_, 0.0)           # (B,H,W)
    G = signal_power / (signal_power + noise_var_ + 1e-12)       # (B,H,W) safe denom

    # Avoid division by zero in the original mag (if any zeros)
    # When mag == 0, set mag to 1 (dummy) and desired magnitude (G*mag) to 0 via G mask.
    zero_mask = (mag == 0)
    safe_mag = mag.copy()
    safe_mag[zero_mask] = 1.0
    # Desired magnitude after filtering is G * mag
    desired_mag = G * mag
    desired_mag[zero_mask] = 0.0

    # Scale complex spectrum to match desired magnitude:
    # im_fft_filt = im_fft * (desired_mag / mag)
    scale = desired_mag / safe_mag                                # (B,H,W)
    im_fft_filt = im_fft * scale                                  # broadcast to complex

    # Inverse FFT and take real part
    im_filt = np.fft.ifft2(im_fft_filt, axes=(-2, -1)).real       # (B,H,W)
    im_filt = im_filt.astype(np.float32, copy=False)

    # Return with original shape
    if not batched:
        return im_filt[0]
    return im_filt

def wiener_dft(im: np.ndarray, sigma: float) -> np.ndarray:
    """
    Adaptive Wiener filter applied to the 2D FFT of the image
    :param im: multidimensional array
    :param sigma: estimated noise power
    :return: filtered version of input im
    """
    noise_var = sigma ** 2
    h, w = im.shape

    im_noise_fft = fft2(im)
    im_noise_fft_mag = np.abs(im_noise_fft / (h * w) ** .5)

    im_noise_fft_mag_noise = wiener_adaptive(im_noise_fft_mag, noise_var)

    zeros_y, zeros_x = np.nonzero(im_noise_fft_mag == 0)

    im_noise_fft_mag[zeros_y, zeros_x] = 1
    im_noise_fft_mag_noise[zeros_y, zeros_x] = 0

    im_noise_fft_filt = im_noise_fft * im_noise_fft_mag_noise / im_noise_fft_mag
    im_noise_filt = np.real(ifft2(im_noise_fft_filt))

    return im_noise_filt.astype(np.float32)


def zero_mean(im: np.ndarray) -> np.ndarray:
    """
    ZeroMean called with the 'both' argument, as from Binghamton toolbox.
    :param im: multidimensional array
    :return: zero mean version of input im
    """
    # Adapt the shape ---
    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    # Subtract the 2D mean from each color channel ---
    ch_mean = im.mean(axis=0).mean(axis=0)
    ch_mean.shape = (1, 1, ch)
    i_zm = im - ch_mean

    # Compute the 1D mean along each row and each column, then subtract ---
    row_mean = i_zm.mean(axis=1)
    col_mean = i_zm.mean(axis=0)

    row_mean.shape = (h, 1, ch)
    col_mean.shape = (1, w, ch)

    i_zm_r = i_zm - row_mean
    i_zm_rc = i_zm_r - col_mean

    # Restore the shape ---
    if im.shape[2] == 1:
        i_zm_rc.shape = im.shape[:2]

    return i_zm_rc


def zero_mean_total(im: np.ndarray) -> np.ndarray:
    """
    ZeroMeanTotal as from Binghamton toolbox.
    :param im: multidimensional array
    :return: zero mean version of input im
    """
    im[0::2, 0::2] = zero_mean(im[0::2, 0::2])
    im[1::2, 0::2] = zero_mean(im[1::2, 0::2])
    im[0::2, 1::2] = zero_mean(im[0::2, 1::2])
    im[1::2, 1::2] = zero_mean(im[1::2, 1::2])
    return im


def rgb2gray(im: np.ndarray) -> np.ndarray:
    """
    RGB to gray as from Binghamton toolbox.
    :param im: multidimensional array
    :return: grayscale version of input im
    """
    rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        im_gray = np.copy(im)
    elif im.shape[2] == 1:
        im_gray = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        im_gray = np.dot(im, rgb2gray_vector)
        im_gray.shape = (w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return im_gray.astype(np.float32)


def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Noise variance theshold as from Binghamton toolbox.
    :param wlet_coeff_energy_avg:
    :param noise_var:
    :return: noise variance threshold
    """
    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2


def wiener_adaptive(x: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
    """
    WaveNoise as from Binghamton toolbox.
    Wiener adaptive flter aimed at extracting the noise component
    For each input pixel the average variance over a neighborhoods of different window sizes is first computed.
    The smaller average variance is taken into account when filtering according to Wiener.
    :param x: 2D matrix
    :param noise_var: Power spectral density of the noise we wish to extract (S)
    :param window_size_list: list of window sizes
    :return: wiener filtered version of input x
    """
    window_size_list = list(kwargs.pop('window_size_list', [3, 5, 7, 9]))

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
    for window_idx, window_size in enumerate(window_size_list):
        avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy,
                                                                  window_size,
                                                                  mode='constant')

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    x = x * noise_var / (coef_var_min + noise_var)

    return x


def inten_scale(im: np.ndarray) -> np.ndarray:
    """
    IntenScale as from Binghamton toolbox
    :param im: type np.uint8
    :return: intensity scaled version of input x
    """

    assert (im.dtype == np.uint8)

    T = 252
    v = 6
    # print(im.shape)
    out = np.exp(-1 * (im.astype(int) - T) ** 2 / v)
    out[im < T] = im[im < T] / T
    # print(out.shape)
    return out.astype(np.uint8)


def saturation(im: np.ndarray) -> np.ndarray:
    """
    Saturation as from Binghamton toolbox
    :param im: type np.uint8
    :return: saturation map from input im
    """
    assert (im.dtype == np.uint8)

    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    if im.max() < 250:
        return np.ones((h, w, ch))

    im_h = im - np.roll(im, (0, 1), (0, 1))
    im_v = im - np.roll(im, (1, 0), (0, 1))
    satur_map = \
        np.bitwise_not(
            np.bitwise_and(
                np.bitwise_and(
                    np.bitwise_and(
                        im_h != 0, im_v != 0
                    ), np.roll(im_h, (0, -1), (0, 1)) != 0
                ), np.roll(im_v, (-1, 0), (0, 1)) != 0
            )
        )

    max_ch = im.max(axis=0).max(axis=0)

    for ch_idx, max_c in enumerate(max_ch):
        if max_c > 250:
            satur_map[:, :, ch_idx] = \
                np.bitwise_not(
                    np.bitwise_and(
                        im[:, :, ch_idx] == max_c, satur_map[:, :, ch_idx]
                    )
                )

    return satur_map


def inten_sat_compact(args):
    """
    Memory saving version of inten_scale followed by saturation. Useful for multiprocessing
    :param args:
    :return: intensity scale and saturation of input
    """
    im = args[0]
    return ((inten_scale(im) * saturation(im)) ** 2).astype(np.float32)


"""
Cross-correlation functions
"""


def crosscorr_2d(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """
    PRNU 2D cross-correlation
    :param k1: 2D matrix of size (h1,w1)
    :param k2: 2D matrix of size (h2,w2)
    :return: 2D matrix of size (max(h1,h2),max(w1,w2))
    """
    assert (k1.ndim == 2)
    assert (k2.ndim == 2)

    max_height = max(k1.shape[0], k2.shape[0])
    max_width = max(k1.shape[1], k2.shape[1])

    k1 -= k1.flatten().mean()
    k2 -= k2.flatten().mean()

    k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1])], mode='constant', constant_values=0)
    k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1])], mode='constant', constant_values=0)

    k1_fft = fft2(k1, )
    k2_fft = fft2(np.rot90(k2, 2), )

    return np.real(ifft2(k1_fft * k2_fft)).astype(np.float32)


def aligned_cc(k1: np.ndarray, k2: np.ndarray) -> dict:
    """
    Aligned PRNU cross-correlation
    :param k1: (n1,nk) or (n1,nk1,nk2,...)
    :param k2: (n2,nk) or (n2,nk1,nk2,...)
    :return: {'cc':(n1,n2) cross-correlation matrix,'ncc':(n1,n2) normalized cross-correlation matrix}
    """

    # Type cast
    k1 = np.array(k1).astype(np.float32)
    k2 = np.array(k2).astype(np.float32)

    ndim1 = k1.ndim
    ndim2 = k2.ndim
    print(k1.shape, k2.shape)
    assert (ndim1 == ndim2)

    k1 = np.ascontiguousarray(k1).reshape(k1.shape[0], -1)
    k2 = np.ascontiguousarray(k2).reshape(k2.shape[0], -1)

    assert (k1.shape[1] == k2.shape[1])

    k1_norm = np.linalg.norm(k1, ord=2, axis=1, keepdims=True)
    k2_norm = np.linalg.norm(k2, ord=2, axis=1, keepdims=True)

    k2t = np.ascontiguousarray(k2.transpose())

    cc = np.matmul(k1, k2t).astype(np.float32)
    ncc = (cc / (k1_norm * k2_norm.transpose())).astype(np.float32)

    return {'cc': cc, 'ncc': ncc}


def aligned_cc_torch(k1: np.ndarray, k2: np.ndarray) -> dict:
    """
    Aligned PRNU cross-correlation, GPU-accelerated with PyTorch.
    Input: 
        k1: (n1, nk) or (n1, nk1, nk2, ...)
        k2: (n2, nk) or (n2, nk1, nk2, ...)
    Output:
        dict with:
          'cc'  : (n1,n2) cross-correlation matrix (float32, numpy)
          'ncc' : (n1,n2) normalized cross-correlation matrix (float32, numpy)
    """
    # ---- Cast numpy → torch, move to GPU ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t1 = torch.from_numpy(np.asarray(k1, dtype=np.float32)).to(device)
    t2 = torch.from_numpy(np.asarray(k2, dtype=np.float32)).to(device)

    # Flatten to (N, D)
    t1 = t1.reshape(t1.shape[0], -1).contiguous()
    t2 = t2.reshape(t2.shape[0], -1).contiguous()

    # ---- Norms ----
    t1_norm = torch.norm(t1, p=2, dim=1, keepdim=True)  # (n1,1)
    t2_norm = torch.norm(t2, p=2, dim=1, keepdim=True)  # (n2,1)

    # ---- Cross-correlation matrix ----
    cc = t1 @ t2.T   # (n1,n2)

    # ---- Normalized cross-correlation ----
    denom = t1_norm * t2_norm.T
    ncc = cc / denom.clamp(min=1e-12)

    # ---- Back to CPU numpy ----
    cc = cc.float().cpu().numpy()
    ncc = ncc.float().cpu().numpy()

    return {"cc": cc, "ncc": ncc}


def pce(cc: np.ndarray, neigh_radius: int = 2) -> dict:
    """
    PCE position and value
    :param cc: as from crosscorr2d
    :param neigh_radius: radius around the peak to be ignored while computing floor energy
    :return: {'peak':(y,x), 'pce': peak to floor ratio, 'cc': cross-correlation value at peak position
    """
    assert (cc.ndim == 2)
    assert (isinstance(neigh_radius, int))

    out = dict()

    max_idx = np.argmax(cc.flatten())
    max_y, max_x = np.unravel_index(max_idx, cc.shape)

    peak_height = cc[max_y, max_x]

    cc_nopeaks = cc.copy()
    cc_nopeaks[max_y - neigh_radius:max_y + neigh_radius, max_x - neigh_radius:max_x + neigh_radius] = 0

    pce_energy = np.mean(cc_nopeaks.flatten() ** 2)

    out['peak'] = (max_y, max_x)
    out['pce'] = (peak_height ** 2) / pce_energy * np.sign(peak_height)
    out['cc'] = peak_height

    return out


"""
Statistical functions
"""
def top_k_accuracy(y_true, y_pred, k=1):
    """
    Computes Top-K accuracy.
    y_true: shape [N], ground truth class indices
    y_pred: shape [N, num_classes], predicted scores/probabilities
    """
    # Get the indices of the top k predictions
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:][:, ::-1]  # shape: [N, k], descending order
    # For Top-1: k=1, for Top-5: k=5
    match_array = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return np.mean(match_array)

def stats(cc: np.ndarray, gt: np.ndarray, ) -> dict:
    """
    Compute statistics
    :param cc: cross-correlation or normalized cross-correlation matrix
    :param gt: boolean multidimensional array representing groundtruth
    :return: statistics dictionary
    """
    assert (cc.shape == gt.shape)
    assert (gt.dtype == np.bool)

    assert (cc.shape == gt.shape)
    assert (gt.dtype == np.bool)
    # print(gt.shape)
    # print(cc.shape)
    top_1_acc = top_k_accuracy(np.argmax(gt, axis=0), cc.T, k=1)
    top_5_acc = top_k_accuracy(np.argmax(gt, axis=0), cc.T, k=5)
    fpr, tpr, th = roc_curve(gt.flatten(), cc.flatten())
    auc_score = auc(fpr, tpr)

    # EER
    eer_idx = np.argmin((fpr - (1 - tpr)) ** 2, axis=0)
    eer = float(fpr[eer_idx])

    outdict = {
        'tpr': tpr,
        'fpr': fpr,
        'th': th,
        'auc': auc_score,
        'eer': eer,
        "top-1-acc": top_1_acc,
        "top-5-acc": top_5_acc,
    }

    return outdict


def gt(l1: list or np.ndarray, l2: list or np.ndarray) -> np.ndarray:
    """
    Determine the Ground Truth matrix given the labels
    :param l1: fingerprints labels
    :param l2: residuals labels
    :return: groundtruth matrix
    """
    l1 = np.array(l1)
    l2 = np.array(l2)

    assert (l1.ndim == 1)
    assert (l2.ndim == 1)

    gt_arr = np.zeros((len(l1), len(l2)), np.bool)

    for l1idx, l1sample in enumerate(l1):
        gt_arr[l1idx, l2 == l1sample] = True

    return gt_arr
