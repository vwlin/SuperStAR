#!/bin/bash

python3 utils/generate_imagenet_variants.py --variant imagenet_cs
python3 utils/generate_imagenet_variants.py --variant imagenet_cp
python3 utils/generate_imagenet_variants.py --variant imagenet_c

mv datasets/cifar100_c/CIFAR-100-C/* datasets/cifar100_c
rm -r datasets/cifar100_c/CIFAR-100-C
python3 utils/generate_cifar100_variants.py --variant cifar100_cs
python3 utils/generate_cifar100_variants.py --variant cifar100_c
python3 utils/repackage_cifar100_variants.py --variant cifar100_cs
python3 utils/repackage_cifar100_variants.py --variant cifar100_c