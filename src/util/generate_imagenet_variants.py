'''
Code to generate Imagenet-CS and ImageNet-CP. Augmented from

D. Hendrycks & T. Dietterich. Benchmarking Neural Network Robustness to Common Corruptions
and Perturbations. ICLR, 2019.

Code source: https://github.com/erichson/NoisyMix/blob/main/src/imagenet_models/resnet.py
'''

# -*- coding: utf-8 -*-

import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
from PIL import Image
import argparse

from local import PATHS

# /////////////// Data Loader ///////////////


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DistortImageFolder(data.Dataset):
    def __init__(self, root, method, severity, variant, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.method = method
        self.severity = severity
        self.variant = variant
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            img = self.method(img, self.severity)
        if self.target_transform is not None:
            target = self.target_transform(target)

        save_path = f'{PATHS[self.variant]}/{self.method.__name__}/{str(self.severity)}/{self.idx_to_class[target]}'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path += path[path.rindex('/'):]

        Image.fromarray(np.uint8(img)).save(save_path, quality=85, optimize=True)

        return 0  # we do not care about returning the data

    def __len__(self):
        return len(self.imgs)


# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian, median
from skimage.exposure import adjust_sigmoid
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)

def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Surrogate Distortions ///////////////

def uniform_noise(x, severity=1):
    c = [0.14, 0.22, 0.32, 0.50, 0.90][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.uniform(size=x.shape, low=-c, high=c), 0, 1) * 255

def median_blur(x, severity=1):
    # note: severities 1-4 haven't been calibrated
    c = [2, 3, 4, 5, 6][severity - 1]

    x = np.array(x) / 255.

    channels = []
    for d in range(3):
        channels.append(median(x[:, :, d], np.ones((c,c))))
    channels = np.array(channels).transpose((1, 2, 0)) # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255

def gamma_change(x, severity=1):
    c = [1.4, 1.7, 2.0, 2.5, 3.0][severity - 1]

    x = np.array(x).astype(np.uint8)
    
    invGamma = 1 / c
    table = [((i / 255) ** invGamma)for i in range(256)]
    table = np.array(table)
    x = cv2.LUT(x, table)

    return np.clip(x, 0, 1) * 255

def gamma_decrease(x, severity=1):
    c = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]

    x = np.array(x).astype(np.uint8)
    
    invGamma = 1 / c
    table = [((i / 255) ** invGamma)for i in range(256)]
    table = np.array(table)
    x = cv2.LUT(x, table)

    return np.clip(x, 0, 1) * 255

def sigmoid_increase(x, severity=1): # increase std dev
    c = [7, 8, 9, 10, 11][severity - 1]

    x = np.array(x) / 255.
    x = adjust_sigmoid(x, gain=c)
    
    return np.clip(x, 0, 1) * 255

def sigmoid(x, severity=1):
    c = [7, 6, 5, 4, 3][severity - 1]

    x = np.array(x) / 255.
    x = adjust_sigmoid(x, gain=c)
    
    return np.clip(x, 0, 1) * 255

# /////////////// Additional ImageNet-C Distortions ///////////////

def none(x, severity=1):
    return x


# /////////////// Composite Distortions ///////////////

def gaussian_noise_brightness(x, severity=1):
    # apply gaussian noise
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    # apply brightness shift
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255

    return x

def gaussian_noise_contrast(x, severity=1):
    # apply gaussian noise
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    # apply contrast shift
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1) * 255

    return x

def gaussian_noise_saturate(x, severity=1):
    # apply gaussian noise
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    # apply saturate shift
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255

    return x

def shot_noise_brightness(x, severity=1):
    # apply shot noise
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

    # apply brightness shift
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255

    return x

def shot_noise_contrast(x, severity=1):
    # apply shot noise
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

    # apply contrast shift
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1) * 255

    return x

def shot_noise_saturate(x, severity=1):
    # apply shot noise
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

    # apply saturate shift
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255

    return x

def impulse_noise_brightness(x, severity=1):
    # apply impulse noise
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    x = np.clip(x, 0, 1) * 255

    # apply brightness shift
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255

    return x

def impulse_noise_contrast(x, severity=1):
    # apply impulse noise
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    x = np.clip(x, 0, 1) * 255

    # apply contrast shift
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1) * 255

    return x

def impulse_noise_saturate(x, severity=1):
    # apply impulse noise
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    x = np.clip(x, 0, 1) * 255

    # apply saturate shift
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255

    return x

def speckle_noise_brightness(x, severity=1):
    # apply speckle noise
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    x =  np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    # apply brightness shift
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255

    return x

def speckle_noise_contrast(x, severity=1):
    # apply speckle noise
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    x =  np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    # apply contrast shift
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1) * 255

    return x

def speckle_noise_saturate(x, severity=1):
    # apply speckle noise
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    x =  np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    # apply saturate shift
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255

    return x


# /////////////// End Distortions ///////////////


# /////////////// Further Setup ///////////////


def save_distorted(method=uniform_noise, variant='imagenet_c'):
    for severity in range(1, 6):
        print(method.__name__, severity)
        distorted_dataset = DistortImageFolder(
            root=PATHS['imagenet'],
            method=method, severity=severity, variant=variant,
            transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224)]))
        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=100, shuffle=False, num_workers=4)

        for _ in distorted_dataset_loader: continue


# /////////////// End Further Setup ///////////////

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--variant", choices=['imagenet_cs','imagenet_c','imagenet_cp'], type=str, help='Variant of ImageNet to generate')
args = parser.parse_args()


# /////////////// Display Results ///////////////
import collections

print('\nUsing ImageNet data')

d = collections.OrderedDict()

if args.variant == 'imagenet_cs':
    d['Uniform Noise'] = uniform_noise
    d['Median Blur'] = median_blur
    d['Gamma Change'] = gamma_change
    d['Gamma Decrease'] = gamma_decrease
    d['Sigmoid Increase'] = sigmoid_increase
    d['Sigmoid'] = sigmoid
elif args.variant == 'imagenet_c':
    d['None'] = none
elif args.variant == 'imagenet_cp':
    d['Gaussian Noise Brightness'] = gaussian_noise_brightness
    d['Gaussian Noise Contrast'] = gaussian_noise_contrast
    d['Gaussian Noise Saturate'] = gaussian_noise_saturate
    d['Shot Noise Brightness'] = shot_noise_brightness
    d['Shot Noise Contrast'] = shot_noise_contrast
    d['Shot Noise Saturate'] = shot_noise_saturate
    d['Impulse Noise Brightness'] = impulse_noise_brightness
    d['Impulse Noise Contrast'] = impulse_noise_contrast
    d['Impulse Noise Saturate'] = impulse_noise_saturate
    d['Speckle Noise Brightness'] = speckle_noise_brightness
    d['Speckle Noise Contrast'] = speckle_noise_contrast
    d['Speckle Noise Saturate'] = speckle_noise_saturate
else:
    raise NotImplementedError

for method_name in d.keys():
    print(f'Generating data for {method_name}...')
    save_distorted(d[method_name], args.variant)