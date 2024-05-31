'''
Label all shifts based on whether they satisfy the desired statistical properties for wasserstein distance.
'''

import torch
import pickle
from torchvision import datasets
import os
import ot
import argparse
import numpy as np
import statsmodels.api as sm

from util.data import ImageFolderWithPaths, prepare_torch_tforms, SEVERITIES, IMAGENETC_SHIFTS, IMAGENETCS_SHIFTS, IMAGENETCP_SHIFTS, CIFAR100C_SHIFTS, CIFAR100CS_SHIFTS
from util.utils import load_resnet, evaluate_resnet, prepare_for_wdist
from util.local import ROOT, PATHS
from env.rewards import wasserstein_distance

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", default = 0, type=int, help='GPU')
parser.add_argument("--seed", default = 0, type=int, help='Random seed')
parser.add_argument("--train_dataset", default='imagenet', choices=['imagenet', 'cifar100'], help='Dataset to classifier on')
parser.add_argument("--net", default='baseline', choices=['baseline', 'augmix', 'noisymix', 'deepaugment', 'deepaugment_augmix', 'puzzlemix'], help='Generate labels based on this classifier')
args = parser.parse_args()

BATCH_SIZE = 1000

if __name__ == '__main__':
    print(args)

    if args.train_dataset == 'cifar100' and args.net in ['deepaugment', 'deepaugment_augmix']:
        raise NotImplementedError

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # set device
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    device_ids = [args.gpu]
    torch.cuda.set_device(args.gpu)

    # load classifier
    resnet = load_resnet(args.train_dataset, args.net, args.gpu)
    
    # set-up for wdist
    print('Creating projection matrix...')
    im_width = {'imagenet':224, 'cifar100':32}[args.train_dataset]
    projection_dim = {'imagenet':5000, 'cifar100':100}[args.train_dataset]
    projection_matrix, hist = prepare_for_wdist(batch_size=BATCH_SIZE, image_dim=im_width*im_width, projection_dim=projection_dim, device=device)
    
    # prepare transforms
    transform, initial_transform, partial_transform, normalize = prepare_torch_tforms(im_width, args.train_dataset, args.net)

    # prepare validation set for wdist
    if args.train_dataset == 'imagenet':
        validation_pth = PATHS['imagenet']
        validation_set = datasets.ImageFolder(validation_pth, transform=transform)
    elif args.train_dataset == 'cifar100':
        validation_set = datasets.CIFAR100(root=f'{ROOT}/datasets', train=False, download=True, transform=transform)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)

    # load previous label data if it exists
    base_dir = f'{ROOT}/decision_tree/{args.train_dataset}_train'
    os.makedirs(base_dir, exist_ok=True)
    for dataset in ['c', 'cs', 'cp']:
        if args.train_dataset == 'cifar100' and dataset == 'cp':
            continue

        data_path = f'{base_dir}/labels_{dataset}.pkl'
        pth_dataset_base = PATHS[f'{args.train_dataset}_{dataset}']

        if args.train_dataset == 'imagenet':
            SHIFTS = {
                'c':IMAGENETC_SHIFTS,
                'cs':IMAGENETCS_SHIFTS,
                'cp':IMAGENETCP_SHIFTS
            }[dataset]
        elif args.train_dataset == 'cifar100':
            SHIFTS = {
                'c':CIFAR100C_SHIFTS,
                'cs':CIFAR100CS_SHIFTS
            }[dataset]
        else:
            raise NotImplementedError

        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                linear_fit_data=pickle.load(f)
        else:
            linear_fit_data = {}

        for shift in SHIFTS:
            if shift in linear_fit_data:
                continue
            if shift == 'none':
                continue # only one severity

            accuracies = [0]*len(SEVERITIES)
            wdists = [0]*len(SEVERITIES)
            for severity in SEVERITIES:
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)

                print('\nProcessing,', shift, severity, '...')

                # load validation set
                validation, _ = next(iter(validation_loader))
                validation = validation.to(device)

                # create shifted dataloader
                pth_dataset = pth_dataset_base + shift + '/' + str(severity)
                data = ImageFolderWithPaths(pth_dataset, transform=initial_transform)
                data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)

                # get accuracies and wasserstein distances
                ct = 0
                with torch.no_grad():
                    for images, targets, _ in data_loader:
                        if ct % 10 == 0:
                            print('Batch ' + str(ct) + ' of ' + str(int(len(data_loader.dataset)/BATCH_SIZE)))
                        ct += 1

                        if args.train_dataset == 'cifar100' and args.net == 'augmix':
                            cropped_resized_normalized = images.detach().clone()
                        else:
                            if shift == 'none':
                                cropped_resized_normalized = normalize(images.detach().clone())
                            else:
                                cropped_resized_normalized = partial_transform(images.detach().clone())

                        # evaluate accuracy
                        cropped_resized_normalized, targets = cropped_resized_normalized.cuda(), targets.cuda()
                        logits = resnet(cropped_resized_normalized)
                        pred = logits.data.max(1)[1]
                        total_correct = pred.eq(targets.data).sum().item()
                        accuracies[severity-1] += total_correct/BATCH_SIZE

                        # evaluate wdist
                        wdist = wasserstein_distance(cropped_resized_normalized, validation, projection_matrix, hist)
                        wdists[severity-1] += wdist

                accuracies[severity-1] /= int(len(data_loader.dataset)/BATCH_SIZE)
                wdists[severity-1] /= int(len(data_loader.dataset)/BATCH_SIZE)
                print('acc:', accuracies[severity-1])
                print('wdist:', wdists[severity-1])

            # get linear fit
            for i in range(len(accuracies)):
                print(f'severity {str(i)}: ', accuracies[i], wdists[i])
            accuracies, wdists = np.array(accuracies), np.array(wdists)

            covs = sm.add_constant(accuracies, prepend=True, has_constant='add')
            linear_fit = sm.OLS(wdists, covs)
            fitted_model = linear_fit.fit()
            print('fitted with parameters', fitted_model.params,'and r2 value', fitted_model.rsquared)

            # save to pickle file
            linear_fit_data[shift] = {
                'accuracis':accuracies,
                'wdists':wdists,
                'parameters':fitted_model.params,
                'rsquared':fitted_model.rsquared
            }
            
            with open(data_path, 'wb') as f:
                pickle.dump(linear_fit_data, f)