'''
Image correction environment. Follows OpenAI gym conventions.
'''
import torch
from torchvision import datasets, transforms as T
from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
from time import time
import os

from env.rewards import wasserstein_distance, similarity, calc_reward
from env.observations import N_STATES, get_observation
from env.actions import N_THREADS, N_MATLAB_THREADS, ACTIONS, apply_action
from util.data import ImageFolderWithPaths, prepare_torch_tforms, IMAGENETCS_SHIFTS_LIMITED, IMAGENETC_SHIFTS, IMAGENETCP_SHIFTS, CIFAR100C_SHIFTS
from util.utils import prepare_for_wdist, prepare_transformation_tools
from util.local import ROOT, PATHS

STEP_LIMIT = 5

class Shift(Env):
    def __init__(self,
                 mode:str = 'train',
                 eval_dataset:str = 'imagenet_c',
                 severities:list[int] = [1,2,3,4,5],
                 batch_size:int = 1000,
                 projection_dim = 5000,
                 lmbda:float = 1,
                 sim_thresh:float = 0.9993,
                 alpha:float = 0.9,
                 gpu:int = 0,
    ):
        if mode not in ['train', 'eval']:
            raise ValueError('Mode must be either train or eval.')
        if eval_dataset not in ['imagenet_c', 'imagenet_cp', 'cifar100_c']:
            raise ValueError('Evaluation dataset must be imagenet_c, imagenet_cp, or cifar100_c.')

        super(Shift, self).__init__()

        # processing details
        self.device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
        self.n_threads = N_THREADS

        # RL details
        self.n_actions = len(ACTIONS)
        self.batch_size = batch_size
        self.projection_dim = projection_dim
        self.mode = mode
        self.eval_dataset = eval_dataset
        if self.mode == 'train':
            self.shifts = IMAGENETCS_SHIFTS_LIMITED
        else:
            self.shifts = {
                'imagenet_c':IMAGENETC_SHIFTS,
                'imagenet_cp':IMAGENETCP_SHIFTS,
                'cifar100_c':CIFAR100C_SHIFTS
            }[self.eval_dataset]
        self.severities = severities
        self.lmbda = lmbda
        self.sim_thresh = sim_thresh
        self.alpha = alpha

        # set observation and action space
        self.observation_space = Box(shape=(N_STATES,), low=-np.inf, high=np.inf)
        self.action_space = Discrete(self.n_actions)

        # prepare for image data
        im_width = {'imagenet_c':224, 'imagenet_cp':224, 'cifar100_c':32}[self.eval_dataset]
        self.image_dim = im_width**2

        print('Preparing validation data...')
        self.transform, self.initial_transform, self.partial_transform, self.normalize = prepare_torch_tforms(im_width, self.eval_dataset.split('_')[0], net='baseline')
        if self.mode == 'train' or 'imagenet' in self.eval_dataset:
            validation_pth = PATHS['imagenet']
            validation_data = datasets.ImageFolder(validation_pth, transform=self.transform)
        else:
            validation_data = datasets.CIFAR100(root=f'{ROOT}/datasets', train=False, download=True, transform=self.transform)
        self.validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=self.batch_size, shuffle=True)

        # create matrix for projection
        print('Creating projection matrix...')
        self.projection_matrix, self.hist = prepare_for_wdist(
            batch_size=self.batch_size,
            image_dim=self.image_dim, projection_dim=self.projection_dim,
            device=self.device)

        # tools for transformations
        print('Preparing for transformations...')
        self.tools = prepare_transformation_tools(N_MATLAB_THREADS)

    def step(self, action:int):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        
        self.history.append(action) # record actions ints to know where to save images

        s = time()
        nanflag, self.images, new_paths = apply_action(action = ACTIONS[action],
                                                        set = self.images, set_paths = self.images_paths,
                                                        history = self.history, dir_name = self.shift + '_' + str(self.severity),
                                                        tools = self.tools, device = self.device, n_threads = self.n_threads,
                                                        dataset = self.eval_dataset.split('_')[0])
        if new_paths is not None:
            self.images_paths = new_paths
        e = time()
        print('applied action', action, 'in', e-s, 's')

        if nanflag == False:
            cropped_resized_normalized = self.partial_transform(self.images.detach().clone())

            if self.mode == 'train' or 'imagenet' in self.eval_dataset:
                self.observation = get_observation(cropped_resized_normalized, self.device, self.n_threads, dataset='imagenet')
            else:
                self.observation = get_observation(cropped_resized_normalized, self.device, self.n_threads, dataset='cifar100')

            wdist = wasserstein_distance(cropped_resized_normalized, self.validation, self.projection_matrix, self.hist)
            sim = similarity(self.shifted, cropped_resized_normalized)
            reward = calc_reward(wdist, sim, self.lmbda, self.sim_thresh)

            print('wasserstein distance: ' + str(wdist))
            print('ssim: ' + str(sim))

            self.step_ct += 1
            done = (wdist <= self.alpha*self.shifted_baseline) or (self.step_ct >= STEP_LIMIT)

            info = {
            'shift':self.shift,
            'severity':self.severity,
            'action':(action, ACTIONS[action]),
            'obs_after_action':self.observation.cpu(),
            'wasserstein_distance':wdist,
            'similarity':sim,
            'reward':reward,
            }

            return self.observation, reward, done, info, None
        else:
            return 0, 0, False, None, nanflag

    def render(self):
        raise NotImplementedError
    
    def reset(self, shift:str = None, severity:int = None):
        if self.mode == 'train':
            if shift is not None:
                print('Shift cannot be deterministically set in train mode. Proceeding with randomly selected shift.')
            if severity is not None:
                print('Severity cannot be deterministically set in train mode. Proceeding with randomly selected severity.')

            self.shift = self.shifts[np.random.randint(len(self.shifts))]
            self.severity = self.severities[np.random.randint(len(self.severities))]
            print("\nSelected level " + str(self.severity) + " " + self.shift + " shift")
            shifted_pth = os.path.join(PATHS['imagenet_cs'], self.shift, str(self.severity))

        else:
            if shift not in self.shifts:
                raise ValueError('Shift must be one of %r.' % self.shifts)
            if severity not in self.severities:
                raise ValueError('Severity must be one of %r.' % self.severities)
            
            self.shift = shift
            self.severity = severity
            shifted_pth = os.path.join(PATHS[self.eval_dataset], self.shift, str(self.severity))

        # load shifted data
        shifted_data = ImageFolderWithPaths(shifted_pth, transform=self.initial_transform)
        shifted_loader = torch.utils.data.DataLoader(shifted_data, batch_size=self.batch_size, shuffle=True)
        self.images, self.images_targets, self.images_paths = next(iter(shifted_loader))
        self.images, self.images_targets, self.images_paths = self.images.to(self.device), self.images_targets.to(self.device), list(self.images_paths)
        self.shifted = self.images.detach().clone()

        # load validation data
        self.validation, self.validation_targets = next(iter(self.validation_loader))
        self.validation, self.validation_targets = self.validation.to(self.device), self.validation_targets.to(self.device)

        # keep track of history of action in an episode
        self.history = []

        # get initial values
        cropped_resized_normalized = self.partial_transform(self.images.detach().clone())
        self.validation_baseline = wasserstein_distance(self.validation, self.validation, self.projection_matrix, self.hist)
        print('validation vs validation wasserstein distance: ' + str(self.validation_baseline))
        self.shifted_baseline = wasserstein_distance(cropped_resized_normalized, self.validation, self.projection_matrix, self.hist) # starting wasserstein distance
        print('shifted vs validation wasserstein distance: ' + str(self.shifted_baseline))

        sim = similarity(self.shifted, cropped_resized_normalized)
        reward = calc_reward(self.shifted_baseline, sim, self.lmbda, self.sim_thresh)

        if self.mode == 'train' or 'imagenet' in self.eval_dataset:
            self.observation = get_observation(cropped_resized_normalized, self.device, self.n_threads, dataset='imagenet')
        else:
            self.observation = get_observation(cropped_resized_normalized, self.device, self.n_threads, dataset='cifar100')

        self.step_ct = 0

        info = {
            'shift':self.shift,
            'severity':self.severity,
            'action':(None,None),
            'obs_after_action':self.observation.cpu(),
            'wasserstein_distance':self.shifted_baseline,
            'similarity':sim,
            'reward':reward,
        }
        
        return self.observation, info

    def set_mode(self, mode:str = 'train'):
        if mode not in {'train', 'eval'}:
            raise ValueError('Mode must be either train or eval.')
        self.mode = mode