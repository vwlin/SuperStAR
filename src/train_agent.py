'''
Train SuperStAR agent.
'''
import argparse
import numpy as np
import torch
import os
import pickle

from env.environment import Shift
from env.actions import ACTIONS
from models.agent import Agent
from util.data import IMAGENETCS_SHIFTS_LIMITED
from util.utils import load_resnet, evaluate_resnet

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", default = 0, type=int, help='GPU')
parser.add_argument("--seed", default = 0, type=int, help='Random seed')
parser.add_argument("--learning_rate", default=0.0001, type=float, help='Learning rate for actor and critic')
parser.add_argument("--learning_freq", default=1, type=int, help='Do updates after this many episodes')
parser.add_argument("--discount", default=0.9, type=float, help='Discount rate')
parser.add_argument("--decay_rate", default=0.07, type=float, help='Rate of exponential decay')
parser.add_argument("--lmbda", default=20.0, type=float, help='Hyperparameter for reward')
parser.add_argument("--sim_thresh", default=0.994, type=float, help='Hyperparameter for reward')
parser.add_argument("--alpha", default=0.9, type=float, help='Hyperparameter for stopping condition')
parser.add_argument("--save_prefix", default = '', type=str, help='Prefix for saved networks and logs')
args = parser.parse_args()

def trainAgent(env, episode_count = 100):
    state_length = env.observation_space.shape[0]
    action_length = env.action_space.n
    exploration_number = 0.9

    # create log dir and path to save logs
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = '_'.join(filter(None, [args.save_prefix, 'train_log.pkl']))
    log_file = os.path.join(log_dir, log_name)

    # create directories for saving actor/critic nets
    network_dir = '_'.join(filter(None, [args.save_prefix, 'networks']))
    if not os.path.exists(network_dir):
        os.makedirs(network_dir)
    actor_file = os.path.join(network_dir, 'actor_network.pth.tar')
    critic_file = os.path.join(network_dir, 'critic_network.pth.tar')

    agent = Agent(state_length, action_length, args.gpu,
                args.learning_rate, args.discount,
                actor_file=actor_file, critic_file=critic_file)
    
    experience = {}
    logs = {'args':args, 'actions':ACTIONS, 'shifts':IMAGENETCS_SHIFTS_LIMITED, 'severities':env.severities}
    total_reward = 0.0
    episode = 0
    accuracy = None
    while episode < episode_count:
        current_episode_transitions = {}
        current_episode_logs = {}
        total_episode_reward = 0.0

        if exploration_number > 0.1:
            exploration_number = pow(0.9, args.decay_rate*(episode+1)) # between 0.0  and 1.0, reaches 0.1 in about 3 hrs (110 ep)
        if exploration_number < 0.1:
            exploration_number = 0.1
        
        print('\n------------------------------------------------------------\nepisode ' + str(episode))
        state, info = env.reset()
        print('initial state:', info['obs_after_action'])
        print('initial reward:', info['reward'])
        current_episode_logs[env.step_ct] = info

        total_episode_reward = 0.0
        current_episode_transitions = {}
        done = False
        while not done:
            print('\nstep ' + str(env.step_ct))
            
            if(np.random.uniform(0,1) <= exploration_number): # explore
                action = np.random.randint(action_length)
                action_distribution, value = agent.actor_net(state), agent.critic_net(state)
                log_prob = action_distribution.log_prob(torch.as_tensor(action).to(agent.device)).unsqueeze(0)
                print('action:', str(action), '-', ACTIONS[action])

                next_state, reward, done, info, nanflag = env.step(action)
                info['mode'] = 'explore'
            else: # exploit
                action_distribution, value = agent.actor_net(state), agent.critic_net(state)
                action = action_distribution.sample()
                log_prob = action_distribution.log_prob(action).unsqueeze(0)
                print('action:', str(action), '-', ACTIONS[action])

                next_state, reward, done, info, nanflag = env.step(action.cpu().numpy())
                info['mode'] = 'exploit'

            # end episode if nan value encountered
            if nanflag:
                break

            print('next state:', info['obs_after_action'])
            print('reward:', info['reward'])

            # save data for training updates
            transition = {
                'state':state,
                'action':action,
                'next_state':info['obs_after_action'],
                'reward':reward,
                'QValue':value,
                'action_log':log_prob
            }
            current_episode_transitions[env.step_ct-1] = transition

            # save data for logging
            current_episode_logs[env.step_ct] = info # note different indexing than current_episode_transitions

            state = next_state
            total_episode_reward += reward

        # if nan value encountered, skip saving information and retry
        if nanflag:
            print('nan image. restarting episode')
            continue

        print()

        experience[episode] = current_episode_transitions
        total_reward += total_episode_reward
        if episode % args.learning_freq == 0:
            actor_loss, critic_loss = agent.update_policy(experience)
            agent.save_to_disk(agent.actor_file, agent.critic_file)
            experience = {}

            current_episode_logs['actor_loss'] = actor_loss.cpu()
            current_episode_logs['critic_loss'] = critic_loss.cpu()

            print("\nAt episode - ", episode , ": Total episode reward = ", total_episode_reward, ", Average reward = ", total_reward / float(args.learning_freq))
            total_reward = 0.0

        logs[episode] = current_episode_logs
        if episode % 10 == 0:
            with open(log_file, 'wb') as f:
                pickle.dump(logs, f)

        episode += 1

if __name__ == '__main__':
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env = Shift(mode = 'train',
                severities = [5],
                batch_size = 1000,
                projection_dim = 5000,
                lmbda = args.lmbda,
                sim_thresh = args.sim_thresh,
                alpha = args.alpha,
                gpu=args.gpu)

    trainAgent(env, 2000)

    env.close()

    