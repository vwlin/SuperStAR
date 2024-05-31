'''
Apply supervisor on top of agent to check for some stopping conditions.
The downstream net is not used by SuperStAR, but it is included in the code to aid tuning.
'''

import argparse
import pickle
import numpy as np
import os

from util.data import IMAGENETC_SHIFTS, IMAGENETCP_SHIFTS, CIFAR100C_SHIFTS, SEVERITIES
from util.local import ROOT

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_prefix", default = '', type=str, help='')
parser.add_argument("--eval_dataset", default='imagenet_c', choices=['imagenet_c', 'imagenet_cp', 'cifar100_c'], help='Dataset to evaluate on')
parser.add_argument("--downstream_net", default='baseline', choices=['baseline', 'augmix', 'noisymix', 'deepaugment', 'deepaugment_augmix', 'puzzlemix'], help='Classifier to test on')
parser.add_argument("--beta", default=0.995, type=float, help='Hyperparameter for stopping condition')
args = parser.parse_args()

if __name__ == '__main__':
    print(args)

    SHIFTS = {
        'imagenet_c':IMAGENETC_SHIFTS,
        'imagenet_cp':IMAGENETCP_SHIFTS,
        'cifar100_c':CIFAR100C_SHIFTS
    }[args.eval_dataset]
    
    # load decision tree
    decision_tree_dir = f"{ROOT}/decision_tree/{args.eval_dataset.split('_')[0]}_train"
    with open(f'{decision_tree_dir}/decision_tree.pkl', 'rb') as f:
        decision_tree = pickle.load(f)

    # open records pickle
    record_dir = f'{ROOT}/logs/{args.eval_dataset}_eval'
    record_pth = os.path.join(record_dir,'_'.join(filter(None, [args.save_prefix, args.downstream_net, 'eval.pkl'])))
    with open(record_pth, 'rb') as f:         
        records = pickle.load(f)

    average_improvements = {}
    final_actions = {}

    # loop through all shifts
    for shift in SHIFTS:
        ave_shift_accuracy = 0.0

        # loop through all severities
        for severity in SEVERITIES:
            print('\n', shift, severity)
            total_severity_acc = 0.0

            # get test_traces of correct shift and severity from records
            key = shift
            if shift != 'none':
                key += '_' + str(severity)
            if key not in records.keys():
                continue
            record = records[key]
            test_traces = record['all']

            num_episodes = len(test_traces)
            action_sequences = {}
            for episode in range(num_episodes):
                print('\nepisode', episode)
                
                episode_traces = test_traces[episode]

                # get baselines
                initial_step_trace = episode_traces['initial']
                baseline_wdist = initial_step_trace['wdist']
                uncleaned_accuracy = initial_step_trace['shift_acc']

                # iterate through all steps in episode
                num_steps = len(episode_traces) - 1
                cleaned_accuracy = uncleaned_accuracy
                actions = []
                unclipped_actions = []
                #
                latest_wdist = baseline_wdist
                stopping_condition = False
                print('uncleaned acc', uncleaned_accuracy)
                print('baseline wdist', baseline_wdist)
                for time in range(num_steps):
                    print('-----')
                    step_trace = episode_traces[time+1]

                    # apply dt abstaining
                    state = step_trace['state'].cpu().numpy().tolist()

                    act_on_batch = decision_tree.predict([state]) # 1 to act, 0 to abstain
                    print('act?', act_on_batch)
                    
                    if act_on_batch == 1:
                        # get cleaned accuracy corresponding to max reward
                        wdist = step_trace['next_wdist']

                        if wdist >= latest_wdist*args.beta:
                            stopping_condition = True

                        if not stopping_condition:
                            latest_wdist = wdist

                            cleaned_accuracy = step_trace['shift_acc']
                            actions.append(step_trace['action'].cpu().numpy())
                    else:
                        stopping_condition = True

                    print('unclipped acc after this step', step_trace['shift_acc'])
                    print('wdist after this step',step_trace['next_wdist'])
                    unclipped_actions.append(step_trace['action'].cpu().numpy())

                print('unclipped actions', unclipped_actions)
                print('selected actions', actions)

                action_sequences[episode] = actions

                # record information
                acc_improvement = cleaned_accuracy - uncleaned_accuracy
                total_severity_acc += acc_improvement

            if shift == 'none':
                print(shift, '- average over', num_episodes, 'trials:', total_severity_acc/num_episodes)
            else:
                print(shift, severity, '- average over', num_episodes, 'trials:', total_severity_acc/num_episodes)

            ave_shift_accuracy += total_severity_acc

            final_actions[key] = action_sequences

            if shift == 'none':
                break # only one severity

        if key in records.keys():
            if shift == 'none':
                ave_shift_accuracy = ave_shift_accuracy /num_episodes
            else:
                ave_shift_accuracy = ave_shift_accuracy / len(SEVERITIES) / num_episodes
            print('***', shift, '- average over', len(SEVERITIES), 'severities:', ave_shift_accuracy, '\n')

            average_improvements[shift] = ave_shift_accuracy

    supervisor_records = {}
    supervisor_records['averange_improvements'] = average_improvements
    supervisor_records['final_actions'] = final_actions

    supervised_record_pth = os.path.join(record_dir,'_'.join(filter(None, [args.save_prefix, args.downstream_net, 'supervised_eval.pkl'])))
    with open(supervised_record_pth, 'wb') as f:
        pickle.dump(supervisor_records, f)

    # print summary
    print('\n\n---------- SUMMARY ----------\n')
    for shift in SHIFTS:
        if shift not in average_improvements.keys():
            continue
        if len(shift) < 8:
            result = shift + '\t\t\t {pct:.2f}%'
        elif len(shift) > 14:
            result = shift + '\t {pct:.2f}%'
        else:
            result = shift + '\t\t {pct:.2f}%'
        print(result.format(pct = average_improvements[shift]*100))