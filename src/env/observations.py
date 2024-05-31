'''
Functions to calculate state representation of current image batch.
'''
import multiprocessing
import torch
import numpy as np
import cv2
import pywt
from skimage.filters.rank import entropy
from skimage.morphology import disk

from util.data import IMAGENET_MEAN, IMAGENET_STD, CIFAR100_MEAN, CIFAR100_STD

N_STATES = 3

# input: image (ndarray, dims [3, 224, 224], normalized)
#        action (int)
# output: 3 values
def _get_dwt_state(image, dataset):
    if dataset == 'imagenet':
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif dataset == 'cifar100':
        mean, std = CIFAR100_MEAN['baseline'], CIFAR100_STD['baseline']

    image = np.moveaxis(image, 0, -1) # move color channels to end

    image = std * image + mean # denormalize image
    image = np.clip(image, 0, 1)* 255 # return to range [0, 255]
    image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = np.float32(image)
    image /= 255

    coeffs2 = pywt.dwt2(image, 'bior1.3')
    LL, (_, _, HH) = coeffs2

    average_brightness = cv2.mean(LL)
    standard_deviation = LL.std()
    sk_image = HH
    entr_img = entropy(sk_image, disk(10))

    return average_brightness[0], standard_deviation, np.mean(entr_img)

def _state_thread_wrapper_queue(input_queue:multiprocessing.Queue,
                               result_queue:multiprocessing.Queue,
                               input_array, dataset):
    while not input_queue.empty():
        try:
            im_idx= input_queue.get(timeout=1)
            image = input_array[im_idx]
            img_state = _get_dwt_state(image, dataset)
            result_queue.put((im_idx, img_state))
        except Exception as e:
            print(e)

# input: array (ndarray, dims [..., 3, 224, 224])
#        threads_count (int)
# output: list of three values
def _run_state_threads_queue(array, dataset, threads_count=40):
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    for index in range(len(array)):
        input_queue.put(index)

    proclist = [multiprocessing.Process(target=_state_thread_wrapper_queue, 
                                        args=(input_queue, output_queue, array, dataset)) for i in range(threads_count)]

    for proc in proclist:
        proc.start()

    procs_alive = True
    result_list = np.zeros((len(array),3))
    while procs_alive:
        try:
            if not output_queue.empty():
                im_idx, state_result = output_queue.get(timeout=1)
                result_list[im_idx, :] = state_result
                # prog_bar.update(n=1)
            any_alive = False
            for proc in proclist:
                if proc.is_alive():
                    any_alive = True
            procs_alive = any_alive
        except Exception as e:
            print(e)

    for proc in proclist:
        proc.join()

    return result_list

# get surrogate state (of dimension size) from set
# input: set (tensor, dims [..., 3, 224, 224])
#        dataset (imagenet or cifar100)
#        device
#        n_threads
def get_observation(set, device, n_threads, dataset='imagenet', mean=True):
    dwt_vals = _run_state_threads_queue(set.cpu().numpy(), dataset, n_threads)
    
    if not mean:
        return dwt_vals

    dwt_vals = torch.FloatTensor(dwt_vals).to(device)
    state = torch.mean(dwt_vals, 0)
    return state