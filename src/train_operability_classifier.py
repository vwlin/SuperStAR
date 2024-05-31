from sklearn import tree
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
parser.add_argument("--seed", default = 0, type=int, help='Random seed')
parser.add_argument("--train_dataset", default='imagenet', choices=['imagenet', 'cifar100'], help='Dataset to classifier on')
parser.add_argument("--net", default='baseline', choices=['baseline', 'augmix', 'noisymix', 'deepaugment', 'deepaugment_augmix', 'puzzlemix'], help='Generate labels based on this classifier')
parser.add_argument("--mode", default = 'test', choices=['test', 'plot'], help='')
args = parser.parse_args()

MODE = args.mode
BATCH_SIZE = 2000
TEST_BATCH_SIZE = 100
TRAIN_VAL_SPLIT = [0.8, 0.2]
N_THREADS = 40
R2_THRESH = {'imagenet':0.4, 'cifar100':0.6}[args.train_dataset]
N_THREADS = 40

BASE_DIR = f'{ROOT}/decision_tree/{args.train_dataset}_train'
DATA_FILE = f'{BASE_DIR}/decision_tree_data.pkl'
MODEL_FILE = f'{BASE_DIR}/decision_tree.pkl'

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    im_width = {'imagenet':224, 'cifar100':32}[args.train_dataset]
    transform, initial_transform, partial_transform, normalize = prepare_torch_tforms(im_width, args.train_dataset, args.net)

    # load linear fit data for imagenet_cs
    with open(f'{BASE_DIR}/labels_cs.pkl', 'rb') as f:
        linear_fit_data = pickle.load(f)

    # load or generate data
    device = torch.device("cpu")
    if not os.path.exists(DATA_FILE):
        print('generating new data')
        X = []
        Y = []
        SHIFTS = {'imagenet':IMAGENETCS_SHIFTS, 'cifar100':CIFAR100CS_SHIFTS}[args.train_dataset]
        for (s,shift) in enumerate(SHIFTS):
            print('\n', shift)
            
            for severity in SEVERITIES[-1:]:
                # get data
                shifted_pth = PATHS[f'{args.train_dataset}_cs'] + shift + '/' + str(severity)
                shifted = ImageFolderWithPaths(shifted_pth, transform=initial_transform)
                shifted_loader = torch.utils.data.DataLoader(shifted, batch_size=BATCH_SIZE, shuffle=True)
                images, _, _ = next(iter(shifted_loader))
                cropped_resized_normalized = partial_transform(images.detach().clone())
                state = get_observation(cropped_resized_normalized, device, N_THREADS, mean=False)
                X += state.tolist()

                # get labels
                slope = linear_fit_data[shift]['parameters'][1]
                r2 = linear_fit_data[shift]['rsquared']
                label = slope < 0 and r2 > R2_THRESH
                print('label:', label)
                Y += [int(label)]*BATCH_SIZE # 1 if satisfies desired property

        data = {'X':X, 'Y':Y}
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(data, f)
    else:
        print('loading generated data')
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        X = data['X']
        Y = data['Y']

    # shuffle and decide split index for train, val
    assert len(X) == len(Y)
    num_train = int(TRAIN_VAL_SPLIT[0]*len(X))
    num_val = len(X) - num_train
    idx = list(range(len(X)))
    random.shuffle(idx)
    X = [X[i] for i in idx]
    Y = [Y[i] for i in idx]

    X_train, Y_train = X[:num_train], Y[:num_train]
    X_val, Y_val = X[num_train:], Y[num_train:]
    assert num_train == len(X_train) == len(Y_train)
    assert num_val == len(X_val) == len(Y_val)

    print('num train:', num_train)
    print('num val:', num_val)

    # load or train model
    if not os.path.exists(MODEL_FILE):
        print('\ntraining new model')
        best_clf = None
        best_acc = 0.
        best_depth = 0
        for max_depth in range(3,11):
            clf = tree.DecisionTreeClassifier(max_depth = max_depth)
            clf = clf.fit(X_train,Y_train)

            predictions = clf.predict(X_val)
            acc = accuracy_score(Y_val, predictions)

            if acc > best_acc:
                best_acc = acc
                best_clf = clf
                best_depth = max_depth

        print('\nselected tree max depth of', best_depth, 'with accuracy:', best_acc, '\n')
        clf = best_clf      

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(clf, f)
    else:
        print('loading trained model')
        with open(MODEL_FILE, 'rb') as f:
            clf = pickle.load(f)

        predictions = clf.predict(X_val)
        acc = accuracy_score(Y_val, predictions)
        print('\naccuracy on shuffled data:', acc, '\n')

    if MODE == 'test':
        SEED = args.seed + 10000
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        print('\ntesting on train shifts')
        SHIFTS = {'imagenet':IMAGENETCS_SHIFTS, 'cifar100':CIFAR100CS_SHIFTS}[args.train_dataset]
        for (s,shift) in enumerate(SHIFTS):
            for severity in SEVERITIES:
                # get data
                shifted_pth = PATHS[f'{args.train_dataset}_cs'] + shift + '/' + str(severity)
                shifted = ImageFolderWithPaths(shifted_pth, transform=initial_transform)
                shifted_loader = torch.utils.data.DataLoader(shifted, batch_size=TEST_BATCH_SIZE, shuffle=True)
                images, _, _ = next(iter(shifted_loader))
                cropped_resized_normalized = partial_transform(images.detach().clone())
                state = get_observation(cropped_resized_normalized, device, N_THREADS, dataset=args.train_dataset, mean=False)
                test_X = state.tolist()

                # get labels
                slope = linear_fit_data[shift]['parameters'][1]
                r2 = linear_fit_data[shift]['rsquared']
                label = slope < 0 and r2 > R2_THRESH
                test_Y = [int(label)]*TEST_BATCH_SIZE # 1 if satisfies desired property

                # get predictions and accuracies
                predictions = clf.predict(test_X)
                acc = accuracy_score(test_Y, predictions)
                print(shift, severity, ' accuracy:', acc)
            print()

        print('\ntesting on test shifts')
        SHIFTS = {'imagenet':IMAGENETC_SHIFTS, 'cifar100':CIFAR100C_SHIFTS}[args.train_dataset]
        for (s,shift) in enumerate(SHIFTS):
            if shift == 'none':
                continue
            for severity in SEVERITIES:

                shifted_pth = PATHS[f'{args.train_dataset}_c'] + shift + '/' + str(severity)
                shifted = ImageFolderWithPaths(shifted_pth, transform=initial_transform)
                shifted_loader = torch.utils.data.DataLoader(shifted, batch_size=TEST_BATCH_SIZE, shuffle=True)
                images, _, _ = next(iter(shifted_loader))
                cropped_resized_normalized = partial_transform(images.detach().clone())
                state = get_observation(cropped_resized_normalized, device, N_THREADS, dataset=args.train_dataset, mean=False)
                test_X = state.tolist()

                predictions = clf.predict(test_X)
                print(shift, severity, ' average prediction:', np.mean(predictions))
            print()

        if args.train_dataset == 'imagenet':
            print('\ntesting on test composite shifts')
            for (s,shift) in enumerate(IMAGENETCP_SHIFTS):
                for severity in SEVERITIES:

                    shifted_pth = PATHS['imagenet_cp'] + shift + '/' + str(severity)
                    shifted = ImageFolderWithPaths(shifted_pth, transform=initial_transform)
                    shifted_loader = torch.utils.data.DataLoader(shifted, batch_size=TEST_BATCH_SIZE, shuffle=True)
                    images, _, _ = next(iter(shifted_loader))
                    cropped_resized_normalized = partial_transform(images.detach().clone())
                    state = get_observation(cropped_resized_normalized, device, N_THREADS, dataset=args.train_dataset, mean=False)
                    test_X = state.tolist()

                    predictions = clf.predict(test_X)
                    print(shift, severity, ' average prediction:', np.mean(predictions))
                print()

    elif MODE == 'plot':
        tree.plot_tree(clf, max_depth=3, fontsize=10)
        plt.show()
    else:
        raise NotImplementedError