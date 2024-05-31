This directory is from the NoisyMix github: https://github.com/erichson/NoisyMix/tree/main/src

This is a quick fix to issues with loading the NoisyMix CIFAR-100 weights. The likely cause is that the whole model was originally saved, rather than the state_dict alone.