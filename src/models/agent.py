import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import sys

def compute_return(trajectories, gamma = 0.99):
    # Assumption, the trajectories are a collection of lists
    # Each list, is in turn a list of tuples : (state, action, reward)

    trajectory_values = {}
    for trajectory_index in trajectories.keys():
        trajectory = trajectories[trajectory_index]
        no_of_steps = len(trajectory)

        trace_value = {}
        return_val = 0
        for step_index in reversed(range(no_of_steps)):
            reward = trajectory[step_index]['reward']
            return_val = reward + gamma * return_val
            trace_value[step_index] = return_val

        trajectory_values[trajectory_index] = trace_value

    return trajectory_values

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

class Agent():
    def __init__(self, input_count, action_count, gpu, learning_rate=0.001, discount=0.99, actor_file = "./networks/actor_network.pkl", critic_file = "./networks/critic_network.pkl"):

        self.actor_file = actor_file
        self.critic_file = critic_file
        self.device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

        self.actor_learning_rate = learning_rate
        self.critic_learning_rate = learning_rate # mimic actor lr
        self.discount_rate = discount

        print(actor_file, critic_file)

        if (os.path.exists(self.actor_file) and os.path.exists(self.critic_file)):
            self.actor_net = Actor(input_count, action_count)
            self.actor_net.load_state_dict(torch.load(self.actor_file, map_location = self.device))
            self.actor_net.to(self.device)
            self.critic_net = Critic(input_count, 1)
            self.critic_net.load_state_dict(torch.load(self.critic_file, map_location = self.device))
            self.critic_net.to(self.device)
            print("Picked the networks from previous iterations.. ")
        else :
            self.actor_net = Actor(input_count, action_count)
            self.actor_net.to(self.device)
            self.critic_net = Critic(input_count, 1)
            self.critic_net.to(self.device)
            print("Built new networks")

        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = self.actor_learning_rate)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = self.critic_learning_rate)

    def update_policy(self, trajectories):

        if not trajectories :
            return

        # Estimate the return from the next states as learnt from the trajectories
        discounted_rewards = compute_return(trajectories, gamma=self.discount_rate)

        no_of_training_iterations = 1
        batch_size = len(trajectories)

        for training_step in range(no_of_training_iterations):
            # Build a batch of traning data
            episode_indices = list(trajectories.keys())
            actor_loss_list = []
            critic_loss_list = []
            entropy_loss_list = []

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()

            # for batch_index in range(batch_size):
            for episode_index in episode_indices:
                time_indices = list(trajectories[episode_index].keys())

                for time_index in time_indices:

                    advantage = trajectories[episode_index][time_index]['QValue'] \
                    - torch.tensor(discounted_rewards[episode_index][time_index])

                    action_log = trajectories[episode_index][time_index]['action_log']
                    actor_loss_list.append(action_log * advantage.clone().detach())
                    critic_loss_list.append(0.5 * advantage.pow(2))

            actor_loss = torch.stack(actor_loss_list, dim = 0).sum()
            critic_loss = torch.stack(critic_loss_list, dim = 0).sum()

            print("Actor loss - ", actor_loss)
            print("Critic loss - ", critic_loss)

            actor_loss.backward()
            critic_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        return actor_loss, critic_loss


    def save_to_disk(self, actor_path, critic_path):

        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)