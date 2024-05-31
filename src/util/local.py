'''
Settings for local use.
'''

ROOT = 'your_path_to/SuperStAR/src'
PATHS = {
    # imagenet datasets
    'imagenet':f'{ROOT}/datasets/imagenet/val/',
    'imagenet_c':f'{ROOT}/datasets/imagenet_c/',
    'imagenet_cs':f'{ROOT}/datasets/imagenet_cs/',
    'imagenet_cp':f'{ROOT}/datasets/imagenet_cp/',

    # imagenet pretrained models
    'imagenet_baseline':f'{ROOT}/checkpoints/imagenet_baseline.pth.tar',
    'imagenet_augmix':f'{ROOT}/checkpoints/imagenet_augmix.pth.tar',
    'imagenet_noisymix':f'{ROOT}/checkpoints/imagenet_noisymix.pth.tar',
    'imagenet_deepaugment':f'{ROOT}/checkpoints/imagenet_deepaugment.pth.tar',
    'imagenet_deepaugment_augmix':f'{ROOT}/checkpoints/imagenet_deepaugment_augmix.pth.tar',
    'imagenet_puzzlemix':f'{ROOT}/checkpoints/imagenet_puzzlemix.pth.tar',

    # cifar100 datasets
    'cifar100_c':f'{ROOT}/datasets/cifar100_c/',
    'cifar100_cs':f'{ROOT}/datasets/cifar100_cs/',

    # cifar100 pretrained models
    'cifar100_noisymix':f'{ROOT}/checkpoints/cifar100_noisymix.pt',
    'cifar100_puzzlemix':f'{ROOT}/checkpoints/cifar100_puzzlemix.pth.tar',

    # matlab working dir
    'matlab': f'{ROOT}/datasets/matlab_working_dir/'
}