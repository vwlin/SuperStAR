from torchvision import datasets, transforms as T
import numpy as np

IMAGENETCS_SHIFTS = ['uniform_noise', 'median_blur', 'gamma_change', 'gamma_decrease', 'sigmoid', 'sigmoid_increase']
IMAGENETCS_SHIFTS_LIMITED = ['uniform_noise', 'gamma_change']
IMAGENETC_SHIFTS = ['none', 'gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise',
                  'defocus_blur', 'gaussian_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                  'brightness', 'contrast', 'saturate', 'fog', 
                  'frost', 'snow', 'spatter', 
                  'jpeg_compression', 'pixelate', 'elastic_transform']
IMAGENETCP_SHIFTS = ['gaussian_noise_brightness', 'gaussian_noise_contrast', 'gaussian_noise_saturate',
                  'impulse_noise_brightness', 'impulse_noise_contrast', 'impulse_noise_saturate',
                  'shot_noise_brightness', 'shot_noise_contrast', 'shot_noise_saturate',
                  'speckle_noise_brightness', 'speckle_noise_contrast', 'speckle_noise_saturate']
                  
CIFAR100CS_SHIFTS = ['uniform_noise', 'median_blur', 'gamma_change', 'gamma_decrease', 'sigmoid', 'sigmoid_increase']
CIFAR100C_SHIFTS = ['none', 'gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise',
                  'defocus_blur', 'gaussian_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                  'brightness', 'contrast', 'saturate', 'fog', 
                  'frost', 'snow', 'spatter', 
                  'jpeg_compression', 'pixelate', 'elastic_transform']

SEVERITIES = [1,2,3,4,5]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR100_MEAN = {
    'baseline':np.array([125.3, 123.0, 113.9]) / 255.0,
    'augmix':[],
    'noisymix':np.array([0.5, 0.5, 0.5]),
    'puzzlemix':np.array([129.3, 124.1, 112.4]) / 255.0
}
CIFAR100_STD = {
    'baseline':np.array([63.0, 62.1, 66.7]) / 255.0,
    'augmix':[],
    'noisymix':np.array([0.5, 0.5, 0.5]),
    'puzzlemix':np.array([68.2, 65.4, 70.4]) / 255.0
}

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def prepare_torch_tforms(im_width, dataset, net='baseline'):
    if dataset == 'imagenet':
        tform_list = [T.ToTensor(),
                    T.Resize(256),
                    T.CenterCrop(im_width),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        transform = T.Compose(tform_list)
        initial_transform = T.ToTensor()
        partial_transform = T.Compose(tform_list[1:])
        normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    elif dataset == 'cifar100':
        mean = CIFAR100_MEAN[net]
        std = CIFAR100_STD[net]
        if net == 'baseline':
            tform_list = [T.ToTensor(),
                        T.Normalize(mean, std)]
            partial_transform = T.Compose(tform_list[1:])
            normalize = T.Compose(tform_list[1:])
        elif net == 'augmix':
            partial_transform = T.Compose([])
            tform_list = [T.ToTensor()]
            normalize = T.Compose([])
        elif net == 'noisymix':
            tform_list = [
                T.ToTensor(),
                T.Normalize(mean, std)]
            partial_transform = T.Compose(tform_list[1:])
            normalize = T.Compose(tform_list[1:])
        elif net == 'puzzlemix':
            tform_list = [T.ToTensor(),
                            T.Normalize(mean, std)]
            partial_transform = T.Compose(tform_list[1:])
            normalize = T.Compose(tform_list[1:])
        else:
            raise NotImplementedError

        transform = T.Compose(tform_list)
        initial_transform = T.ToTensor()
        
    else:
        raise NotImplementedError

    return transform, initial_transform, partial_transform, normalize