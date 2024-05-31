#!/bin/bash

python3 label_operability.py --train_dataset imagenet --net baseline
python3 train_operability_classifier.py --train_dataset imagenet --net baseline
python3 eval_operability_classifier.py --train_dataset imagenet --net baseline

python3 label_operability.py --train_dataset cifar100 --net baseline
python3 train_operability_classifier.py --train_dataset cifar100 --net baseline
python3 eval_operability_classifier.py --train_dataset cifar100 --net baseline