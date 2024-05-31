'''
Functions to apply actions.
'''
import os
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as TF
import multiprocessing

from env.transformations import bilateral_denoise, wavelet_denoise, clahe
from util.local import PATHS

N_THREADS = 40
N_MATLAB_THREADS = 20

ACTIONS = [
    'nop',
    'nn_denoiser',
    'bilateral_2', # filter size 2
    'wavelet_bayes', # Bayes Shrink
    'wavelet_visu_2', # Visu Shrink, divider 2
    'clahe_2_1', # grid size 2, clip limit 1
    'clahe_2_2', # grid size 2, clip limit 2
    'clahe_6_1', # grid size 6, clip limit 1
    ]

# input: image (ndarray, dims [3, 224, 224], normalized)
#        action (int)
# output: image (ndarray, dims [3, 224, 224], not normalized, range [0,1])
def _transform_images(image, transform, tools, split_tforms = True):
    image = np.moveaxis(image, 0, -1) # move color channels to end

    if not split_tforms: # loaded images were normalized, so they need to be denormalized
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean # denormalize image
        image = np.clip(image, 0, 1)

    if transform == 'bilateral_2':
        image = bilateral_denoise(image, 2)

    elif transform == 'wavelet_bayes':
        image = wavelet_denoise(image, 'BayesShrink')

    elif transform == 'wavelet_visu_2':
        image = wavelet_denoise(image, 'VisuShrink', 2)

    elif transform == 'clahe_2_1':
        clahe_tool = tools['clahe_21']
        image = clahe(image, clahe_tool)

    elif transform == 'clahe_2_2':
        clahe_tool = tools['clahe_22']
        image = clahe(image, clahe_tool)
        
    elif transform == 'clahe_6_1':
        clahe_tool = tools['clahe_61']
        image = clahe(image, clahe_tool)
        
    else:
        raise NotImplementedError
        
    image = np.moveaxis(image, -1, 0) # move color channels to beginning

    if(np.isnan(image).any()):
        print('nan value found after applying action')

    return image

def _action_thread_wrapper_queue(input_queue:multiprocessing.Queue,
                                 result_queue:multiprocessing.Queue,
                                 input_array, action, tools):
    while not input_queue.empty():
        try:
            im_idx = input_queue.get(timeout=1)
            image = input_array[im_idx]
            trans_image = _transform_images(image, action, tools)
            result_queue.put((im_idx, trans_image))
        except Exception as e:
            print(e)

# input: array (ndarray, dims [..., 3, 224, 224])
#        action (int)
#        threads_count (int)
# output: list of transformed images (ndarray, dims [3, 244, 244], not normalized, range [0,1])
def _run_action_threads_queue(array, action, tools, threads_count=80):
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    for index in range(len(array)):
        input_queue.put(index)

    proclist = [multiprocessing.Process(target=_action_thread_wrapper_queue, 
                                        args=(input_queue, output_queue, array, action, tools)) for i in range(threads_count)]

    for proc in proclist:
        proc.start()

    procs_alive = True
    result_list = np.zeros(array.shape)
    while procs_alive:
        try:
            if not output_queue.empty():
                im_idx, transform_result = output_queue.get(timeout=1)
                result_list[im_idx,:,:,:] = transform_result
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

# apply transformation to set and return transformed set
# input: transformation (int)
#        set (tensor, dims [..., 3, 224, 224])
def apply_action(action, set, set_paths, history, dir_name, tools, device, n_threads, dataset='imagenet'):

    if action == 'nop':
        return False, set, None
    elif action == 'nn_denoiser':
        # save previous step's images (not cropped, not resized, not normalized) to file system
        prev_step_paths = set_paths
        if len(history) > 1: # previous step is NOT unprocessed images
            prev_step_dir = f"{PATHS['matlab']}/{dataset}/{dir_name}/actions"
            for a in history[:-1]: # exclude current action (that is the NN denoiser)
                prev_step_dir += '_' + str(a)
            prev_step_dir += '/'
            
            os.makedirs(prev_step_dir, exist_ok=True)
            for i in range(len(prev_step_paths)):
                prev_step_paths[i] = prev_step_dir + os.path.basename(set_paths[i])
                if not os.path.exists(prev_step_paths[i]):
                    torchvision.utils.save_image(set[i], prev_step_paths[i])

        # construct base path for saving images
        out_dir = f"{PATHS['matlab']}/{dataset}/{dir_name}/actions"
        for a in history:
            out_dir += '_' + str(a)
        out_dir += '/'

        # process images
        eng, net = tools['matlab_eng'], tools['denoise_net']
        if dataset == 'imagenet':
            out_paths = eng.action_nn_denoiser(net, prev_step_paths, out_dir)
        elif dataset == 'cifar100':
            # since cifar images are so small, they must be resized larger before passing through the denoiser
            # they are resized back to 32x32 after denoising
            out_paths = eng.action_nn_denoiser_resize(net, prev_step_paths, out_dir)
        else:
            raise NotImplementedError

        # load matlab-processed images
        transformed = set.cpu().detach().clone()
        for i in range(len(out_paths)):
            # https://discuss.pytorch.org/t/how-to-read-just-one-pic/17434
            loaded_image = Image.open(out_paths[i])
            loaded_image = TF.to_tensor(loaded_image)
            loaded_image.unsqueeze_(0)
            transformed[i] = loaded_image

        # load matlab-processed images (speedup for when images are already saved)
        # transformed = set.cpu().detach().clone()
        # images_paths = list(images_paths)
        # for i in range(len(prev_step_paths)):
        #     basenm = os.path.basename(prev_step_paths[i])
        #     new_pth = out_dir + basenm
        #     images_paths[i] = new_pth ### ?

        #     # https://discuss.pytorch.org/t/how-to-read-just-one-pic/17434
        #     loaded_image = Image.open(new_pth)
        #     loaded_image = TF.to_tensor(loaded_image)
        #     loaded_image.unsqueeze_(0)
        #     transformed[i] = loaded_image
        # ### TODO: integrate above
        
        transformed = transformed.to(device)
        images_paths = out_paths

        return False, transformed, images_paths
    else:
        transformed = _run_action_threads_queue(set.cpu().numpy(), action, tools, n_threads)

        nanflag = np.isnan(transformed).any()

        transformed = torch.FloatTensor(transformed).to(device)

        return nanflag, transformed, None