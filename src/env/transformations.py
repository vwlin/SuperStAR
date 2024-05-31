'''
Image cleaning transformations.
'''
import cv2
import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma, denoise_bilateral

def bilateral_denoise(image, win_size):
    image = denoise_bilateral(image,
                              win_size=win_size, sigma_color=None, sigma_spatial=1, multichannel=True)
    return image

def wavelet_denoise(image, method, divider=1):
    if method == 'VisuShrink':
        sigma_est = estimate_sigma(image, multichannel=True, average_sigmas=True)
        image = denoise_wavelet(image, convert2ycbcr=True,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est/divider, rescale_sigma=True, multichannel=True)
    elif method == 'BayesShrink':
        image = denoise_wavelet(image, convert2ycbcr=True,
                                  method='BayesShrink', mode='soft',
                                  rescale_sigma=True, multichannel=True)
    else:
        raise NotImplementedError
    return image

def clahe(image, clahe_tool):
    image = image*255
    image = image.astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[...,0] = clahe_tool.apply(lab[...,0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = image/255
    return image