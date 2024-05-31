'''
Calculation of reward.
'''

import numpy as np
import torch
from torchvision import transforms as T
import ot
from pytorch_msssim import ssim

# estimate the wasserstein distance between set1 and set2, using projection matrix V and hist
# input: set1 (tensor, dims [..., 1, 224, 224])
#        set2 (tensor, dims [..., 1, 224, 224])
#        V (tensor, dims [224*224, projection_dim])
#        hist (ndarray, dims [validation_size])
def wasserstein_distance(set1, set2, V, hist):
    # convert to grayscale
    set1 = T.Grayscale().forward(set1)
    set2 = T.Grayscale().forward(set2)
    
    # flatten
    set1 = torch.flatten(set1, start_dim=1)
    set2 = torch.flatten(set2, start_dim=1)

    # reduce dimensions
    dr_set1 = torch.matmul(set1, V).cpu().numpy()
    dr_set2 = torch.matmul(set2, V).cpu().numpy()

    # compute wasserstein distance
    M = ot.dist(dr_set1, dr_set2, metric='euclidean')
    w1 = ot.emd2(hist, hist, M, numItermax=500000)

    return w1

# get the similarity between two batches of images
# input: set1 (tensor, dims [..., 1, 224, 224] normalized)
#        set2 (tensor, dims [..., 1, 224, 224] normalized)
def similarity(set1, set2):
    invTrans = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                            T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                        ])
    sim = ssim(invTrans(set1), invTrans(set2), data_range=255, nonnegative_ssim=True, size_average=True)
    torch.cuda.empty_cache()
    return sim.cpu().numpy()

# get the reward from wasserstein distance
# input: wdist as np, sim as np
def calc_reward(wdist, sim, lmbda, sim_thresh):
    sim_loss = np.log(1-sim)
    if sim_loss <= np.log(1-sim_thresh):
        sim_loss = 0
    return -wdist + lmbda * sim_loss