'''
Converts CIFAR-100-C and CIFAR-100-CS from numpy files to a directory tree
'''

import numpy as np
import os
from PIL import Image
import argparse

from data import CIFAR100C_SHIFTS, CIFAR100CS_SHIFTS
from local import PATHS

'''
Label list from: https://github.com/knjcode/cifar2png/blob/master/common/preprocess.py
To avoid a label generation mismatch, must name folders alphabetically so sorting is correct.
See: https://github.com/pytorch/vision/blob/af3077e3d0c3537476ccc2ed3f29a45e56ed30ee/torchvision/datasets/folder.py#L35
'''
CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--variant", choices=['cifar100_cs', 'cifar100_c'], type=str, help='Variant of ImageNet to generate')
args = parser.parse_args()

if args.variant == 'cifar100_cs':
    SHIFTS = CIFAR100CS_SHIFTS
elif args.variant == 'cifar100_c':
    SHIFTS = CIFAR100C_SHIFTS
else:
    raise NotImplementedError

for shift in SHIFTS:
    print('----------', shift, '----------')
    print()
    all_data = np.load(f'{PATHS[args.variant]}/{shift}.npy')
    all_labels = np.load(f'{PATHS[args.variant]}/labels.npy')
    for severity in range(1,6):
        print(shift, severity)
        data = all_data[(severity-1)*10000:severity*10000,:,:,:].astype('uint8')
        labels = all_labels[(severity-1)*10000:severity*10000]
        
        base_dir = os.path.join(PATHS[args.variant], shift, str(severity))
        for i in range(data.shape[0]):
            save_dir = os.path.join(base_dir, CIFAR100_LABELS_LIST[labels[i]])
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, str(i)+'.PNG')

            im = Image.fromarray(data[i,:,:,:])
            im = im.convert('RGB')
            im.save(save_path)