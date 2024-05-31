#!/bin/bash

python3 eval_agent.py --eval_dataset imagenet_c --alpha 0.9
python3 apply_supervisor.py --eval_dataset imagenet_c --beta 0.995
for net in baseline augmix noisymix deepaugment deepaugment_augmix puzzlemix
do
    python3 eval_downstream_models.py --eval_dataset imagenet_c --downstream_net $net --supervised_records_pth logs/imagenet_c_eval/baseline_supervised_eval.pkl
done

python3 eval_agent.py --eval_dataset imagenet_cp --alpha 0.9
python3 apply_supervisor.py --eval_dataset imagenet_cp --beta 0.995
for net in baseline augmix noisymix deepaugment deepaugment_augmix puzzlemix
do
    python3 eval_downstream_models.py --eval_dataset imagenet_cp --downstream_net $net --supervised_records_pth logs/imagenet_cp_eval/baseline_supervised_eval.pkl
done

python3 eval_agent.py --eval_dataset cifar100_c --alpha 0.9
python3 apply_supervisor.py --eval_dataset cifar100_c --beta 0.972
for net in baseline augmix noisymix puzzlemix
do
    python3 eval_downstream_models.py --eval_dataset cifar100_c --downstream_net $net --supervised_records_pth logs/cifar100_c_eval/baseline_supervised_eval.pkl
done