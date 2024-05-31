'''
Utility code.
'''

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torchvision import models
import sys
import matplotlib.pyplot as plt
import matlab.engine
import cv2
from torchdistill.models.classification.wide_resnet import wide_resnet28_10
from robustbench.utils import load_model

from util.local import PATHS
from models.imagenet_puzzlemix import ResNet as imagenet_puzzlemix
from models.imagenet_noisymix import resnet50 as imagenet_noisymix
from models.cifar100_puzzlemix import wrn28_10 as cifar100_puzzlemix
from models.cifar100_noisymix import wideresnet28 as cifar100_noisymix

def relu(x):
    return x * (x > 0)

def moving_mode(a, n=50) :
    a = np.array(a).flatten()
    modes = []
    for i in range(len(a) - n + 1):
        mode, _ = stats.mode(a[i:i+n])
        modes.append(mode[0])
    return modes

def plot_mode_action(dicty, dictx=None, window=20):
    if dictx is not None:
        assert dictx.keys() == dicty.keys(), 'dict keys for x and y must match'

    n_keys = len(dicty.keys())
    assert n_keys > 1, 'dict must include more than 1 signal. to plot a single signal, use plot_with_moving_average()'

    fig, ax = plt.subplots(n_keys,1)
    for (i, key) in enumerate(dicty.keys()):
        signal = dicty[key]
        mode_signal = moving_mode(signal, window)

        if dictx is not None:
            idx = dictx[key]
            ax[i].scatter(idx[:-(window-1)], mode_signal)
        else:
            ax[i].scatter(mode_signal)
        ax[i].grid()
        ax[i].set_ylabel(key)

    return fig, ax

def moving_average(a, n=50) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_dict_with_moving_average(dicty, dictx=None, window=20):
    if dictx is not None:
        assert dictx.keys() == dicty.keys(), 'dict keys for x and y must match'

    n_keys = len(dicty.keys())
    assert n_keys > 1, 'dict must include more than 1 signal. to plot a single signal, use plot_with_moving_average()'

    fig, ax = plt.subplots(n_keys,1)
    for (i, key) in enumerate(dicty.keys()):
        signal = dicty[key]
        ave_signal = moving_average(signal, window)

        if dictx is not None:
            idx = dictx[key]
            ax[i].plot(idx, signal)
            ax[i].plot(idx[:-(window-1)], ave_signal)
        else:
            ax[i].plot(signal)
            ax[i].plot(ave_signal)
        ax[i].grid()
        ax[i].set_ylabel(key)

    return fig, ax

def plot_with_moving_average(y, x = None, window=20):
    fig, ax = plt.subplots()
    ave_y = moving_average(y, window)
    if x is not None:
        ax.plot(x,y)
        ax.plot(x[:-(window-1)], ave_y)
    else:
        ax.plot(y)
        ax.plot(ave_y)
    ax.grid()

    return fig, ax

def _load_imagenet_resnet(dataset, net_name, gpu):
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    device_ids = [gpu]

    if net_name == 'baseline':
        resnet50 = models.resnet50(pretrained=False)
        resnet50.load_state_dict(torch.load(PATHS[f'{dataset}_{net_name}']))
        resnet50 = torch.nn.DataParallel(resnet50, device_ids = device_ids).cuda()
    elif net_name == 'augmix':
        resnet50 = models.resnet50(pretrained=False)
        resnet50 = torch.nn.DataParallel(resnet50, device_ids = device_ids).cuda()
        checkpoint = torch.load(PATHS[f'{dataset}_{net_name}'], map_location=device)
        resnet50.load_state_dict(checkpoint['state_dict'])
    elif net_name == 'noisymix':
        resnet50 = imagenet_noisymix(num_classes=1000)
        resnet50 = torch.nn.DataParallel(resnet50, device_ids = device_ids).cuda()
        checkpoint = torch.load(PATHS[f'{dataset}_{net_name}'], map_location=device)
        resnet50.module.load_state_dict(checkpoint)
    elif net_name == 'deepaugment':
        resnet50 = models.resnet50(pretrained=False)
        resnet50 = torch.nn.DataParallel(resnet50, device_ids = device_ids).cuda()
        checkpoint = torch.load(PATHS[f'{dataset}_{net_name}'], map_location=device)
        resnet50.load_state_dict(checkpoint['state_dict'])
    elif net_name == 'deepaugment_augmix':
        resnet50 = models.resnet50(pretrained=False)
        resnet50 = torch.nn.DataParallel(resnet50, device_ids = device_ids).cuda()
        checkpoint = torch.load(PATHS[f'{dataset}_{net_name}'], map_location=device)
        resnet50.load_state_dict(checkpoint['state_dict'])
    elif net_name == 'puzzlemix':
        # https://github.com/snu-mllab/PuzzleMix/blob/master/imagenet/test.py
        resnet50 = imagenet_puzzlemix('imagenet', 50, 1000, 'bottleneck')
        checkpoint = torch.load(PATHS[f'{dataset}_{net_name}'], map_location=device)
        resnet50.load_state_dict(checkpoint)
        resnet50 = torch.nn.DataParallel(resnet50, device_ids = device_ids).cuda()
    else:
        raise NotImplementedError

    resnet50.to(device)
    return resnet50.eval()

def _load_cifar100_resnet(dataset, net_name, gpu):
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    device_ids = [gpu]

    if net_name == 'baseline':
        wrn = wide_resnet28_10(num_classes=100, pretrained=True)
    elif net_name == 'augmix':
        wrn = load_model(model_name='Hendrycks2020AugMix_WRN', dataset='cifar100', threat_model='corruptions')
    elif net_name == 'noisymix':
        wrn = torch.load(PATHS[f'{dataset}_{net_name}']).to(torch.device('cpu'))
        wrn = torch.nn.DataParallel(wrn, device_ids = device_ids).cuda()
    elif net_name == 'puzzlemix':
        # args here: https://github.com/snu-mllab/PuzzleMix/blob/e2dbf3a2371026411d5741d129f46bf3eb3d3465/main.py#L542
        wrn = cifar100_puzzlemix(num_classes=100)
        wrn = torch.nn.DataParallel(wrn, device_ids = device_ids).cuda()
        wrn.load_state_dict(torch.load(PATHS[f'{dataset}_{net_name}'])['state_dict'])
    else:
        raise NotImplementedError

    wrn.to(device)
    return wrn.eval()

def load_resnet(dataset, net_name, gpu):
    if dataset == 'imagenet':
        return _load_imagenet_resnet(dataset, net_name, gpu)
    elif dataset == 'cifar100':
        return _load_cifar100_resnet(dataset, net_name, gpu)
    else:
        raise NotImplementedError
    
def evaluate_resnet_metrics(resnet, images, targets):
    logits = resnet(images)
    loss = F.cross_entropy(logits, targets)
    pred = logits.data.max(1)[1]
    total_loss = float(loss.data)
    total_correct = pred.eq(targets.data).sum().item()

    return total_correct, total_loss

# evaluate resnet only on a single batch
def evaluate_resnet(resnet, images, targets, batch_size):
    with torch.no_grad():        
        total_correct, _ = evaluate_resnet_metrics(resnet, images, targets)

    accuracy = total_correct * 100 / batch_size
    return accuracy

def prepare_for_wdist(batch_size, image_dim, projection_dim, device):
    shape = torch.Size((image_dim, projection_dim)) # for reduced QR, matrix must be tall
    ortho_matrix = torch.FloatTensor(shape).to(device)
    torch.rand(shape, out=ortho_matrix)
    (projection_matrix, _) = torch.linalg.qr(ortho_matrix) # result is fixed nxd column orthonormal matrix
    
    hist = np.ones((batch_size,))/batch_size

    return projection_matrix, hist

def prepare_transformation_tools(n_matlab_threads):
    eng = matlab.engine.start_matlab()
    denoise_net = eng.denoisingNetwork('DnCNN')
    eng.parpool('Processes', n_matlab_threads)

    clahe_21 = cv2.createCLAHE(clipLimit=1, tileGridSize=(2,2))
    clahe_22 = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))
    clahe_61 = cv2.createCLAHE(clipLimit=1, tileGridSize=(6,6))

    tools = {
        'matlab_eng':eng,
        'denoise_net':denoise_net,
        'clahe_21':clahe_21,
        'clahe_22':clahe_22,
        'clahe_61':clahe_61,
    }
    return tools