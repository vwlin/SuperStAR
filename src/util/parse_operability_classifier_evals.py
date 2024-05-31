'''
Read evaluation pickle file for operability classifier and print results.
'''

# from sklearn import tree
import numpy as np
import pickle
import random
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse

from data import SEVERITIES, IMAGENETC_SHIFTS, IMAGENETCS_SHIFTS, IMAGENETCP_SHIFTS, CIFAR100C_SHIFTS, CIFAR100CS_SHIFTS
from local import ROOT

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_dataset", default='imagenet', choices=['imagenet', 'cifar100'], help='Dataset to classifier on')
args = parser.parse_args()

BATCH_SIZE = 2000
EVAL_FILE = f"{ROOT}/decision_tree/{args.train_dataset}_train/decision_tree_eval.pkl"

if __name__ == '__main__':
    with open(EVAL_FILE, 'rb') as f:
        eval_results = pickle.load(f)

    for dataset in ['c', 'cs', 'cp']:
        if args.train_dataset == 'cifar100' and dataset == 'cp':
            continue

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

        # load data
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
                Y = eval_results[key]['Y']
                predictions = eval_results[key]['predictions']
                predictions_proba = eval_results[key]['probs']
                
                # exclude first batch of severity 5 (this was train/val/data) from surrogate data to calculate accuracy
                if dataset == 'cs' and severity == 5:
                    acc = accuracy_score(Y[BATCH_SIZE:], predictions[BATCH_SIZE:])
                    total_Y += Y[BATCH_SIZE:]
                    total_Yhat += list(predictions[BATCH_SIZE:])
                    total_Yhat_probs += list(predictions_proba[BATCH_SIZE:,1])
                else:
                    acc = accuracy_score(Y, predictions)
                    total_Y += Y
                    total_Yhat += list(predictions[:])
                    total_Yhat_probs += list(predictions_proba[:,1])
                
                ave_pred += list(predictions)
                ave_acc += acc
                
            print('AVERAGES OVER ALL SHIFT')
            print('average predictions:',  np.mean(ave_pred))
            print('accuracy:', ave_acc / 5, '\n')

        print('RESULTS OVER WHOLE TEST')
        acc = accuracy_score(total_Y, total_Yhat)
        print('acc:', acc)
        if len(np.unique(total_Y)) > 1: # must have multiple classes present in labels to perform auroc test
            auroc = roc_auc_score(total_Y, total_Yhat_probs)
            print('auroc:', auroc)