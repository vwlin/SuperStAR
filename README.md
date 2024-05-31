# SuperStAR_L4DC_2024
This is the repository for the paper *[DC4L: Distribution Shift Recovery via Data-Driven Control for Deep Learning Models](https://arxiv.org/abs/2302.10341)*, accepted at L4DC 2024.

*Deep neural networks have repeatedly been shown to be non-robust to the uncertainties of the real world, even to naturally occurring ones. A vast majority of current approaches have focused on data-augmentation methods to expand the range of perturbations that the classifier is exposed to while training. A relatively unexplored avenue that is equally promising involves sanitizing an image as a preprocessing step, depending on the nature of perturbation. In this paper, we propose to use control for learned models to recover from distribution shifts online. Specifically, our method applies a sequence of semantic-preserving transformations to bring the shifted data closer in distribution to the training set, as measured by the Wasserstein distance. Our approach is to 1\) formulate the problem of distribution shift recovery as a Markov decision process, which we solve using reinforcement learning, 2\) identify a minimum condition on the data for our method to be applied, which we check online using a binary classifier, and 3\) employ dimensionality reduction through orthonormal projection to aid in our estimates of the Wasserstein distance. We provide theoretical evidence that orthonormal projection preserves characteristics of the data at the distributional level. We apply our distribution shift recovery approach to the ImageNet-C benchmark for distribution shifts, demonstrating an improvement in average accuracy of up to 14.21% across a variety of state-of-the-art ImageNet classifiers. We further show that our method generalizes to composites of shifts from the ImageNet-C benchmark, achieving improvements in average accuracy of up to 9.81%. Finally, we test our method on CIFAR-100-C and report improvements of up to 8.25%.*

## Prerequisites

### Package Installations
Our code requires python 3.9.12 and MATLAB R2022b with the Deep Learning Toolbox.

1. Follow [instructions](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) to install MATLAB Enging API for Python
2. Install [RobustBench](https://robustbench.github.io/).
    ```
    pip install git+https://github.com/RobustBench/robustbench.git@v1.1
    ```
3. Install remaining requirements.
    ```
    pip install -r requirements.txt
    ```

### Datasets
1. Update `ROOT` in `utils/local.py` to match your local directory tree.
2. Download the [ImageNet](https://www.image-net.org/index.php) 2012 validation set and unpack it into `src/datasets/imagenet/`.
3. Download [ImageNet-C](https://zenodo.org/records/2235448) and unpack it into `src/datasets/imagenet_c/`.
4. Download [CIFAR-100-C](https://zenodo.org/records/3555552) and unpack it into `src/datasets/cifar100_c/`.
5. Generate remaining required data.
    ```
    ./generate_datasets.sh
    ```
Step 5 generates the following:
- ImageNet corrupted by surrogate transformations, under `src/datasets/imagenet_cs/`
- ImageNet corrupted by composite transformations, under `src/datasets/imagenet_cp/`
- CIFAR-100-C corrupted by surrogate transformations, under `src/datasets/cifar100_cs/`
- A "none" shift for ImageNet-C and CIFAR-100-C, under the appropriate folders

Note that when generating corrupted data, some distribution shift may occur due to the loading and saving of data alone, even before a corruption is applied. To ensure consistency across all our datasets, we generate data under "no" shift by passing ImageNet (or CIFAR-100) data through this same load and save process, witholding other corruption. In our evaluations, we use this "none" shift data instead of the original uncorrupted ImageNet (or CIFAR-100) data.

### Weights for Downstream Classifiers
We evaluate downstream classifiers using pre-trained weights provided by the original publication authors. Please note that for PuzzleMix and NoisyMix, the format of these weight files cannot be read using the version of torch we suggest, a byproduct of the versioning requirements necessary for Python-MATLAB integration. We have manually resaved these weights, unchanged, in a format appropriate for our version requirements. We provide the correctly formatted weight files for all classifiers.

Download [pre-trained ImageNet and CIFAR-100 weights](https://drive.google.com/file/d/1CXvBl8K_8889ok3EQ71u_mj3VbhVqcEE/view?usp=sharing) and unpack them to `src/checkpoints`.

## Reproducing our Results
1. Train agent to correct ImageNet corruptions. Pre-trained actor and critic networks are located under `published_results/networks`. To train new networks, run
    ```
    python train_agent.py
    ```
2. Train the operability classifier. Pre-generated operability labels and a pre-trained operability classifier (decision tree) are located under `published_results/decision_tree`. To generate new labels, train a new classifier, and evaluate the trained classifier for ImageNet and CIFAR-100 corruptions, run:
    ```
    ./prepare_operability_classifier.sh
    ```
3. Evaluate SuperStAR on ImageNet-C, ImageNet-CP, and CIFAR-100-C. To select actions and test the accuracies of downstream models for each dataset, run:
    ```
    ./evaluate_superstar.sh
    ```

## BibTeX Entry
```
@article{lin2023dc4l,
  title={DC4L: Distribution Shift Recovery via Data-Driven Control for Deep Learning Models},
  author={Lin, Vivian and Jang, Kuk Jin and Dutta, Souradeep and Caprio, Michele and Sokolsky, Oleg and Lee, Insup},
  journal={arXiv preprint arXiv:2302.10341},
  year={2023}
}
```