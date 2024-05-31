from sklearn import tree
from sklearn.metrics import roc_auc_score
import torch
from torchvision import datasets
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import random
import os
import matplotlib.pyplot as plt
import scipy
import argparse

from util.data import ImageFolderWithPaths, prepare_torch_tforms, SEVERITIES, IMAGENETC_SHIFTS, IMAGENETCS_SHIFTS, IMAGENETCP_SHIFTS, CIFAR100C_SHIFTS, CIFAR100CS_SHIFTS
from util.local import ROOT, PATHS
from env.observations import get_observation

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_dataset", default='imagenet', choices=['imagenet', 'cifar100'], help='Dataset to classifier on')
parser.add_argument("--net", default='baseline', choices=['baseline', 'augmix', 'noisymix', 'deepaugment', 'deepaugment_augmix', 'puzzlemix'], help='Generate labels based on this classifier')
parser.add_argument("--seed", default = 0, type=int, help='Random seed')
args = parser.parse_args()

BATCH_SIZE = 2000
N_THREADS = 40
R2_THRESH = 0.4

BASE_DIR = f'{ROOT}/decision_tree/{args.train_dataset}_train'
DATA_FILE = f'{BASE_DIR}/decision_tree_data.pkl'
MODEL_FILE = f'{BASE_DIR}/decision_tree.pkl'
EVAL_FILE = f'{BASE_DIR}/decision_tree_eval.pkl'

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    im_width = {'imagenet':224, 'cifar100':32}[args.train_dataset]
    transform, initial_transform, partial_transform, normalize = prepare_torch_tforms(im_width, args.train_dataset, 'baseline')

    # load trained model
    print('loading trained model')
    with open(MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)

    # do evaluation
    eval_results = {}
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE, 'rb') as f:
            eval_results=pickle.load(f)

    device = torch.device("cpu")
    for dataset in ['c', 'cs', 'cp']:
        if args.train_dataset == 'cifar100' and dataset == 'cp':
            continue

        data_path = f'{BASE_DIR}/labels_' + dataset + '.pkl'
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
        
        # load linear fit data for imagenet_cs
        with open(data_path, 'rb') as f:
            linear_fit_data = pickle.load(f)

        # load data
        print('generating data and evaluating')
        total_Y = []
        total_Yhat = []
        total_Yhat_probs = []
        for (s,shift) in enumerate(SHIFTS):
            if shift == 'none':
                continue

            print('\n----------', shift, '----------')   
            ave_pred = []
            ave_acc = 0
            for severity in SEVERITIES:
                print(shift, severity)

                key = shift + '_' + str(severity)
                if key in eval_results:
                    continue

                shifted_pth = pth_dataset_base + shift + '/' + str(severity)
                shifted = ImageFolderWithPaths(shifted_pth, transform=initial_transform)
                shifted_loader = torch.utils.data.DataLoader(shifted, batch_size=BATCH_SIZE, shuffle=True)
                
                X = []
                Y = []
                for i,(images, targets, _) in enumerate(shifted_loader):                
                    # get data
                    cropped_resized_normalized = partial_transform(images.detach().clone())
                    state = get_observation(cropped_resized_normalized, device, N_THREADS, mean=False)
                    X += state.tolist()

                    # get labels
                    slope = linear_fit_data[shift]['parameters'][1]
                    r2 = linear_fit_data[shift]['rsquared']
                    label = slope < 0 and r2 > R2_THRESH
                    Y += [int(label)]*BATCH_SIZE # 1 if satisfies desired property

                assert len(X) == len(Y)
                print('num test:', len(X))

                print()
                predictions = clf.predict(X)
                predictions_proba = clf.predict_proba(X)
                print('average predictions on data:', np.mean(predictions))
                
                # exclude first batch of severity 5 (this was train/val/data) from surrogate data to calculate accuracy
                if dataset == 'cs' and severity == 5 and i == 0:
                    acc = accuracy_score(Y[BATCH_SIZE:], predictions[BATCH_SIZE:])
                    total_Y += Y[BATCH_SIZE:]
                    total_Yhat += list(predictions[BATCH_SIZE:])
                    total_Yhat_probs += list(predictions_proba[BATCH_SIZE:,1])
                else:
                    acc = accuracy_score(Y, predictions)
                    total_Y += Y
                    total_Yhat += list(predictions[:])
                    total_Yhat_probs += list(predictions_proba[:,1])
                print('accuracy on data:', acc)

                eval_results[key] = {
                    'X':X,
                    'Y':Y,
                    'accuracy':acc,
                    'predictions':predictions,
                    'probs':predictions_proba
                }
                ave_pred += list(predictions)
                ave_acc += acc

                with open(EVAL_FILE, 'wb') as f:
                    pickle.dump(eval_results, f)

            print('AVERAGES OVER ALL SHIFT')
            print('average predictions:',  np.mean(ave_pred))
            print('accuracy:', ave_acc / 5, '\n')

        acc = accuracy_score(total_Y, total_Yhat)
        auroc = roc_auc_score(total_Y, total_Yhat_probs)
        print('RESULTS OVER WHOLE TEST')
        print('acc:', acc)
        print('auroc:', auroc)