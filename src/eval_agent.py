'''
Evaluate SuperStAR agent.
The downstream net is not used by SuperStAR at this stage, but it is included in the code to aid tuning.
'''

import argparse
import numpy as np
import torch
import pickle
import os

from env.environment import Shift
from models.agent import Agent
from util.data import SEVERITIES, IMAGENETC_SHIFTS, IMAGENETCP_SHIFTS, CIFAR100C_SHIFTS
from util.utils import load_resnet, evaluate_resnet
from util.local import ROOT

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", default = 0, type=int, help='GPU')
parser.add_argument("--seed", default = 0, type=int, help='Random seed')
parser.add_argument("--eval_dataset", default='imagenet_c', choices=['imagenet_c', 'imagenet_cp', 'cifar100_c'], help='Dataset to evaluate on')
parser.add_argument("--downstream_net", default='baseline', choices=['baseline', 'augmix', 'noisymix', 'deepaugment', 'deepaugment_augmix', 'puzzlemix'], help='Classifier to test on')
parser.add_argument("--test_episodes", default = 3, type=int, help='Number of episodes to test over')
parser.add_argument("--lmbda", default=20.0, type=float, help='Hyperparameter for reward')
parser.add_argument("--sim_thresh", default=0.994, type=float, help='Hyperparameter for reward')
parser.add_argument("--alpha", default=0.9, type=float, help='Hyperparameter for stopping condition')
parser.add_argument("--save_prefix", default = '', type=str, help='Prefix for saved networks and logs')
args = parser.parse_args()

def testAgent(env, agent, eval_dataset, downstream_net, shift, severity, test_episodes):
    total_val_acc = 0
    initial_accuracies = []
    final_accuracies = []
    test_traces = {}
    for episode in range(test_episodes) :
        episode_traces = {}
        print('\n------------------------------------------------------------\nepisode ' + str(episode))
        state, info = env.reset(shift, severity)            
        done = False

        # evaluate accuracy on validation set
        val_acc = evaluate_resnet(downstream_net, env.validation, env.validation_targets, env.batch_size)
        print(f'Accuracy on validation set: {str(val_acc)}')
        total_val_acc += val_acc

        # evaluate accuracy on shifted, uncorrected set
        if eval_dataset == 'cifar100' and downstream_net == 'augmix':
            cropped_resized_normalized = images.detach().clone()
        else:
            if env.shift == 'none':
                cropped_resized_normalized = env.normalize(env.images.detach().clone())
            else:
                cropped_resized_normalized = env.partial_transform(env.images.detach().clone())
        init_acc = evaluate_resnet(downstream_net, cropped_resized_normalized, env.images_targets, env.batch_size)
        print(f'Accuracy on shifted set: {str(init_acc)}')
        initial_accuracies.append(init_acc)

        step_traces = {}
        step_traces['state'] = state
        step_traces['wdist'] = info['wasserstein_distance']
        step_traces['sim'] = info['similarity']
        step_traces['reward'] = info['reward']
        step_traces['val_acc'] = val_acc
        step_traces['shift_acc'] = init_acc
        episode_traces['initial'] = step_traces

        while not done:
            print(f'\ntime: {str(env.step_ct)}')
            print(f'state: {state}')

            # select action and apply
            action_distribution = agent.actor_net(state)
            action = action_distribution.sample()
            log_prob = action_distribution.log_prob(action).unsqueeze(0)
            next_state, reward, done, info, _ = env.step(action.cpu().numpy())

            print(f'action: {str(action)}')
            print(f'reward: {str(reward)}')

            # record for downstream evaluation
            step_traces = {}
            step_traces['state'] = state
            step_traces['action'] = action
            step_traces['next_state'] = next_state
            step_traces['next_wdist'] = info['wasserstein_distance'] # after action
            step_traces['next_sim'] = info['similarity'] # after action
            step_traces['reward'] = info['reward'] # after action
            
            state = next_state
            
            # evaluate accuracy on shifted, corrected set at this step
            if eval_dataset == 'cifar100' and downstream_net == 'augmix':
                cropped_resized_normalized = images.detach().clone()
            else:
                if env.shift == 'none':
                    cropped_resized_normalized = env.normalize(env.images.detach().clone())
                else:
                    cropped_resized_normalized = env.partial_transform(env.images.detach().clone())
            final_acc = evaluate_resnet(downstream_net, cropped_resized_normalized, env.images_targets, env.batch_size)
            print('Accuracy on shifted set: ' + str(final_acc))

            step_traces['shift_acc'] = final_acc
            episode_traces[env.step_ct] = step_traces

        final_accuracies.append(final_acc)

        test_traces[episode] = episode_traces

    print('----------SUMMARY----------')
    print('Average accuracy on validation set: ' + str(total_val_acc/test_episodes))
    for i in range(test_episodes):
        print(f'Episode {str(i)}: initial {str(initial_accuracies[i])}, final {str(final_accuracies[i])}')

    return total_val_acc, initial_accuracies, final_accuracies, test_traces

if __name__ == '__main__':
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    SHIFTS = {
        'imagenet_c':IMAGENETC_SHIFTS,
        'imagenet_cp':IMAGENETCP_SHIFTS,
        'cifar100_c':CIFAR100C_SHIFTS
    }[args.eval_dataset]

    projection_dim = {'imagenet_c':5000, 'imagenet_cp':5000, 'cifar100_c':100}[args.eval_dataset]
    env = Shift(mode = 'eval',
                eval_dataset = args.eval_dataset,
                severities = SEVERITIES,
                batch_size = 1000,
                projection_dim = projection_dim,
                lmbda = args.lmbda,
                sim_thresh = args.sim_thresh,
                alpha = args.alpha,
                gpu=args.gpu)

    # load trained agent
    state_length = env.observation_space.shape[0]
    action_length = env.action_space.n
    network_dir = '_'.join(filter(None, [args.save_prefix, 'networks']))
    actor_file = os.path.join(network_dir, 'actor_network.pth.tar')
    critic_file = os.path.join(network_dir, 'critic_network.pth.tar')

    agent = Agent(state_length, action_length, args.gpu,
                actor_file = actor_file, critic_file = critic_file)

    # load downstream model to evaluate accuracy on
    downstream_net = load_resnet(args.eval_dataset.split('_')[0], args.downstream_net, args.gpu)

    records = {}
    record_dir = f'{ROOT}/logs/{args.eval_dataset}_eval'
    os.makedirs(record_dir, exist_ok=True)
    record_pth = os.path.join(record_dir,'_'.join(filter(None, [args.save_prefix, args.downstream_net, 'eval.pkl'])))
    for shift in SHIFTS:
        for severity in SEVERITIES:
            print('\no-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o\n')
            print('shift ' + shift + ', severity ' + str(severity) + '\n')

            # set seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            # run test
            record = {}
            record['total_vacc'], record['initial_accuracies'], record['final_accuracies'], record['all'] = testAgent(
                env = env, agent = agent,
                eval_dataset = args.eval_dataset, downstream_net = downstream_net,
                shift = shift, severity = severity,
                test_episodes = args.test_episodes
            )
            
            # save to records and dump
            key = shift
            if shift != 'none':
                key = shift + '_' + str(severity)
            records[key] = record

            with open(record_pth, 'wb') as f:
                pickle.dump(records, f)

            if shift == 'none':
                break # only one severity

    env.close()

    # print all results
    print('\no-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o\n')
    print('SUMMARY')
    print('\no-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o\n')

    print('---------- none shift ---------')
    print('Average accuracy on validation set: ' + str(records['none']['total_vacc']/test_episodes))
    for i in range(test_episodes):
        print('Episode ' + str(i) + ': initial ' + str(records['none']['initial_accuracies'][i]) + ', final ' + str(records['none']['final_accuracies']))

    for shift in SHIFTS:
        for severity in SEVERITIES:
            key = shift + '_' + str(severity)
            print('---------- shift ' + shift + ', severity ' + str(severity) + ' ----------\n')
            print('Average accuracy on validation set: ' + str(records[key]['total_vacc']/test_episodes))
            for i in range(test_episodes):
                print('Episode ' + str(i) + ': initial ' + str(records[key]['initial_accuracies'][i]) + ', final ' + str(records[key]['final_accuracies']))