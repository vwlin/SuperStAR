'''
Apply actions to corrupted data and evaluate accuracy of downstream model.
'''

import argparse
import pickle
import numpy as np
import os
import torch
import torch.nn.functional as F

from env.actions import N_THREADS, N_MATLAB_THREADS, ACTIONS, apply_action
from util.data import ImageFolderWithPaths, prepare_torch_tforms, SEVERITIES, IMAGENETC_SHIFTS, IMAGENETCP_SHIFTS, CIFAR100C_SHIFTS
from util.utils import prepare_transformation_tools, load_resnet, evaluate_resnet_metrics
from util.local import ROOT, PATHS

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", default = 0, type=int, help='GPU')
parser.add_argument("--seed", default = 0, type=int, help='Random seed')
parser.add_argument("--eval_dataset", default='imagenet_c', choices=['imagenet_c', 'imagenet_cp', 'cifar100_c'], help='Dataset to evaluate on')
parser.add_argument("--downstream_net", default='baseline', choices=['baseline', 'augmix', 'noisymix', 'deepaugment', 'deepaugment_augmix', 'puzzlemix'], help='Classifier to test on')
parser.add_argument("--supervised_records_pth", default='logs/imagenet_c_eval/baseline_supervised_eval.pkl', type=str, help='Load SuperStAR action selections from here')
parser.add_argument("--save_prefix", default = '', type=str, help='')
args = parser.parse_args()

BATCH_SIZE = 1000

if __name__ == '__main__':
    print(args)

    if args.eval_dataset == 'cifar100_c' and args.downstream_net == 'noisymix' and args.gpu != 0:
        raise ValueError('Due to the way the original noisymix wide resnet weights '
            'were saved, the model cannot be evaluated on any GPU other than GPU 0. '
            'This error is a quick fix until the weight file is fixed.')

    if args.eval_dataset == 'cifar100' and args.downstream_net in ['deepaugment', 'deepaugment_augmix']:
        raise NotImplementedError

    SHIFTS = {
        'imagenet_c':IMAGENETC_SHIFTS,
        'imagenet_cp':IMAGENETCP_SHIFTS,
        'cifar100_c':CIFAR100C_SHIFTS
    }[args.eval_dataset]

    # set gpu
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    device_ids = [args.gpu]
    torch.cuda.set_device(args.gpu)

    # tools for transformations
    tools = prepare_transformation_tools(N_MATLAB_THREADS)

    # load downstream model
    resnet = load_resnet(args.eval_dataset.split('_')[0], args.downstream_net, args.gpu)

    # torch transformations for datasets
    im_width = {'imagenet_c':224, 'imagenet_cp':224, 'cifar100_c':32}[args.eval_dataset]
    transform, initial_transform, partial_transform, normalize = prepare_torch_tforms(im_width, args.eval_dataset.split('_')[0], net=args.downstream_net)

    # open supervised records pickle
    with open(args.supervised_records_pth, 'rb') as f:
        supervisor_records=pickle.load(f)
    final_actions = supervisor_records['final_actions']

    final_improvements = {}

    # loop through all shifts
    all_shifts = {}
    record_dir = f'{ROOT}/logs/{args.eval_dataset}_eval'
    record_base_pth = os.path.join(record_dir,'_'.join(filter(None, [args.save_prefix, args.downstream_net])))
    for shift in SHIFTS:
        ave_shift_accuracy_improvement = 0.0
        ave_shift_accuracy_recovered = 0.0
        ave_shift_accuracy_unclean = 0.0
        max_shift_accuracy_recovered = 0.0
        max_shift_accuracy_improvement = -100.00
        max_shift_accuracy_unclean = 0.0
        min_shift_accuracy_recovered = 100.0
        min_shift_accuracy_improvement = 100.00
        min_shift_accuracy_unclean = 100.0

        # loop through all severities
        all_severities_of_shift = {}
        for severity in SEVERITIES:
            severity_data = {}

            # set seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            print('Processing,', shift, severity, '...')

            # load data for this shift and severity
            pth_dataset = PATHS[args.eval_dataset] + shift + '/' + str(severity)
            data = ImageFolderWithPaths(pth_dataset, transform=initial_transform)
            data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)

            # get baseline accuracy on shifted set
            print('Getting baseline accuracies...')
            total_correct = 0
            ct = 0
            with torch.no_grad():
                for images, targets, _ in data_loader:
                    if ct % 10 == 0:
                        print('Batch ' + str(ct) + ' of ' + str(int(len(data_loader.dataset)/BATCH_SIZE)))
                    ct += 1

                    if args.eval_dataset == 'cifar100' and args.downstream_net == 'augmix':
                        cropped_resized_normalized = images.detach().clone()
                    else:
                        if shift == 'none':
                            cropped_resized_normalized = normalize(images.detach().clone())
                        else:
                            cropped_resized_normalized = partial_transform(images.detach().clone())
                    cropped_resized_normalized, targets = cropped_resized_normalized.cuda(), targets.cuda()

                    total_correct += evaluate_resnet_metrics(resnet, cropped_resized_normalized, targets)[0]

                    del images, cropped_resized_normalized, targets
                    torch.cuda.empty_cache()

            uncleaned_accuracy = total_correct / len(data_loader.dataset)
            print('Baseline accuracy,', uncleaned_accuracy)
            severity_data['unclean_acc'] = uncleaned_accuracy
            ave_shift_accuracy_unclean += uncleaned_accuracy

            if uncleaned_accuracy > max_shift_accuracy_unclean:
                max_shift_accuracy_unclean = uncleaned_accuracy
            if uncleaned_accuracy < min_shift_accuracy_unclean:
                min_shift_accuracy_unclean = uncleaned_accuracy

            # get accuracies with correction
            total_severity_acc_improvement = 0.0
            total_severity_acc_recovered = 0.0

            key = shift
            if shift != 'none':
                key += '_' + str(severity)
            if key not in final_actions.keys():
                continue

            action_sequences = final_actions[key]
            num_episodes = len(action_sequences)

            seen_action_sequences = [] # track if there are any duplicate sequences, so we don't have to re-run the trial
            seen_accuracies = []

            # loop over num_episodes trials
            cleaned_accs = []
            for episode in range(num_episodes):
                actions = action_sequences[episode]

                # record if this action sequence is a repeat from a previous episode (allows for some processing speedup)
                seen = 0
                for i in range(len(seen_action_sequences)):
                    if seen_action_sequences[i] == actions:
                        seen += 1
                        idx = i
                        break

                print('Episode', episode, 'actions:', actions)
                if len(actions) == 0:
                    # if no actions, the cleaned accuracy is the same as the uncleaned accuracy
                    print('no actions. continue')  

                    accuracy = uncleaned_accuracy

                    acc_improvement = accuracy - uncleaned_accuracy
                    total_severity_acc_improvement += acc_improvement

                    total_severity_acc_recovered += accuracy

                    seen_action_sequences.append(actions)
                    seen_accuracies.append(accuracy)
                    print('Episode', episode, '-\t', accuracy, '\t', acc_improvement)

                    if accuracy > max_shift_accuracy_recovered:
                        max_shift_accuracy_recovered = accuracy
                    if accuracy < min_shift_accuracy_recovered:
                        min_shift_accuracy_recovered = accuracy

                    if acc_improvement > max_shift_accuracy_improvement:
                        max_shift_accuracy_improvement = acc_improvement
                    if acc_improvement < min_shift_accuracy_improvement:
                        min_shift_accuracy_improvement = acc_improvement

                    cleaned_accs.append(accuracy)

                    continue

                elif seen == 1:
                    # if this action sequence has been seen before, the cleaned accuracy is the same as before
                    print('repeat sequence. continue')

                    accuracy = seen_accuracies[idx]

                    acc_improvement = accuracy - uncleaned_accuracy
                    total_severity_acc_improvement += acc_improvement

                    total_severity_acc_recovered += accuracy
                    print('Episode', episode, '-\t', accuracy, '\t', acc_improvement)

                    if accuracy > max_shift_accuracy_recovered:
                        max_shift_accuracy_recovered = accuracy
                    if accuracy < min_shift_accuracy_recovered:
                        min_shift_accuracy_recovered = accuracy

                    if acc_improvement > max_shift_accuracy_improvement:
                        max_shift_accuracy_improvement = acc_improvement
                    if acc_improvement < min_shift_accuracy_improvement:
                        min_shift_accuracy_improvement = acc_improvement

                    cleaned_accs.append(accuracy)

                    continue

                else:
                    # if this is a new action sequence, apply it and evaluate the cleaned accuracy
                    print('new sequence. do')
                    num_steps = len(actions)

                    total_correct = 0
                    ct = 0
                    with torch.no_grad():
                        for images, targets, paths in data_loader:
                            images, targets = images.to(device), targets.to(device)
                            if ct % 10 == 0:
                                print('Batch ' + str(ct) + ' of ' + str(int(len(data_loader.dataset)/BATCH_SIZE)))
                            ct += 1

                            # apply processing
                            history = []
                            for a in actions:
                                history.append(a)
                                # _, images = apply_transform(a, images, history, paths)
                                nanflag, images, new_paths = apply_action(action = ACTIONS[a],
                                                                            set = images, set_paths = paths,
                                                                            history = history, dir_name = shift + '_' + str(severity),
                                                                            tools = tools, device = device, n_threads = N_THREADS,
                                                                            dataset = args.eval_dataset.split('_')[0])
                                if new_paths is not None:
                                    paths = new_paths
                                
                            # evaluate accuracy
                            if args.eval_dataset == 'cifar100' and args.downstream_net == 'augmix':
                                cropped_resized_normalized = images.detach().clone()
                            else:
                                if shift == 'none':
                                    cropped_resized_normalized = normalize(images.detach().clone())
                                else:
                                    cropped_resized_normalized = partial_transform(images.detach().clone())

                            total_correct += evaluate_resnet_metrics(resnet, cropped_resized_normalized, targets)[0]

                            del images, cropped_resized_normalized, targets
                            torch.cuda.empty_cache()

                    accuracy = total_correct / len(data_loader.dataset)
                    cleaned_accs.append(accuracy)

                    # update list of seen actions and accuracies
                    seen_action_sequences.append(actions)
                    seen_accuracies.append(accuracy)
                    acc_improvement = accuracy - uncleaned_accuracy
                    total_severity_acc_improvement += acc_improvement
                    total_severity_acc_recovered += accuracy

                    if accuracy > max_shift_accuracy_recovered:
                        max_shift_accuracy_recovered = accuracy
                    if accuracy < min_shift_accuracy_recovered:
                        min_shift_accuracy_recovered = accuracy

                    if acc_improvement > max_shift_accuracy_improvement:
                        max_shift_accuracy_improvement = acc_improvement
                    if acc_improvement < min_shift_accuracy_improvement:
                        min_shift_accuracy_improvement = acc_improvement

                    print('Episode', episode, '-\t', accuracy, '\t', acc_improvement)

            severity_data['clean_acc'] = cleaned_accs
            all_severities_of_shift[str(severity)] = severity_data

            if shift == 'none':
                print(shift, '- average over', num_episodes, 'trials:', total_severity_acc_improvement/num_episodes, '\n')
            else:
                print(shift, severity, '- average over', num_episodes, 'trials:', total_severity_acc_improvement/num_episodes, '\n')

            ave_shift_accuracy_improvement += total_severity_acc_improvement
            ave_shift_accuracy_recovered += total_severity_acc_recovered

            if shift == 'none':
                break

        if key in final_actions.keys():
            if shift == 'none':
                ave_shift_accuracy_improvement = ave_shift_accuracy_improvement / num_episodes
                ave_shift_accuracy_recovered = ave_shift_accuracy_recovered / num_episodes
            else:
                ave_shift_accuracy_improvement = ave_shift_accuracy_improvement / len(SEVERITIES) / num_episodes
                ave_shift_accuracy_recovered = ave_shift_accuracy_recovered / len(SEVERITIES) / num_episodes
                ave_shift_accuracy_unclean = ave_shift_accuracy_unclean / len(SEVERITIES)
            print('***', shift, '- average over', len(SEVERITIES), 'severities:', ave_shift_accuracy_improvement)

            final_improvements[shift] = (ave_shift_accuracy_unclean, min_shift_accuracy_unclean, max_shift_accuracy_unclean,
                                        ave_shift_accuracy_recovered, min_shift_accuracy_recovered, max_shift_accuracy_recovered,
                                        ave_shift_accuracy_improvement, min_shift_accuracy_improvement, max_shift_accuracy_improvement)

            all_shifts[shift] = all_severities_of_shift   

        with open(f'{record_base_pth}_final_eval.pkl', 'wb') as f:
            pickle.dump(final_improvements, f)

        with open(f'{record_base_pth}_final_accs_dt.pkl', 'wb') as f:
            pickle.dump(all_shifts, f)

        print()

        # print summary
        print('\n\n---------- RUNNING SUMMARY ----------\n')
        print('SHIFT\t\t\tUNCLEANED ACCURACY \t CLEANED ACCURACY \t IMPROVEMENT')
        for shift in SHIFTS:
            if shift not in final_improvements.keys():
                continue
            
            ave_uncleaned_accuracy, min_uncleaned_accuracy, max_uncleaned_accuracy, ave_cleaned_accuracy, min_cleaned_accuracy, max_cleaned_accuracy, ave_improvement, min_improvement, max_improvement = final_improvements[shift]

            if len(shift) < 8:
                result = shift + '\t\t\t {a_unclean:.2f} ({mn_unclean:.2f}/{mx_unclean:.2f})% \t {a_clean:.2f} ({mn_clean:.2f}/{mx_clean:.2f})% \t {a_improvement:.2f} ({mn_improvement:.2f}/{mx_improvement:.2f})%'
            elif len(shift) > 14:
                result = shift + '\t {a_unclean:.2f} ({mn_unclean:.2f}/{mx_unclean:.2f})% \t {a_clean:.2f} ({mn_clean:.2f}/{mx_clean:.2f})% \t {a_improvement:.2f} ({mn_improvement:.2f}/{mx_improvement:.2f})%'
            else:
                result = shift + '\t\t {a_unclean:.2f} ({mn_unclean:.2f}/{mx_unclean:.2f})% \t {a_clean:.2f} ({mn_clean:.2f}/{mx_clean:.2f})% \t {a_improvement:.2f} ({mn_improvement:.2f}/{mx_improvement:.2f})%'
            print(result.format(a_unclean = ave_uncleaned_accuracy*100, mn_unclean = min_uncleaned_accuracy*100, mx_unclean = max_uncleaned_accuracy*100,
                                a_clean = ave_cleaned_accuracy*100, mn_clean = min_cleaned_accuracy*100, mx_clean = max_cleaned_accuracy*100,
                                a_improvement = ave_improvement*100, mn_improvement = min_improvement*100, mx_improvement = max_improvement*100))

        print()