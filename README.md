# Collaborative Learning with Disentangled Features for Zero-shot Domain Adaptation

[ICCV 2021] This repository is the PyTorch implementation of [**DF-ZSDA**](http://https://openaccess.thecvf.com/content/ICCV2021/html/Jhoo_Collaborative_Learning_With_Disentangled_Features_for_Zero-Shot_Domain_Adaptation_ICCV_2021_paper.html)

> **Collaborative Learning with Disentangled Features for Zero-shot Domain Adaptation**<br>
> Won Young Jhoo, and Jae-Pil Heo <br> 
> Sungkyunkwan University


This code is an experimental version for MNIST/EMNIST/Fashion-MNIST experiments and did not be fully optimized and refactored yet.
You can further improve performance by modifying some parts of the training code I guess.


## Requirements
 * torch == 1.10.0
 * torchvision == 0.11.1
 * Pillow == 7.2.0
 * python-opencv
 * tqdm


## Preparing Datasets

1. **Download BSDS500 Dataset**

We need [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) dataset for synthesizing the color domain (**C**).

Download the dataset and unzip it to the 'data/BSDS500' folder.

2. **Generate 4 Synthetic Domains**

We used Gray(**G**), Color (**C**), Edge (**E**), and Negative (**N**) domain for our experiments.

`sh dataset_gen.sh`

Run this script will generate each domain dataset in the 'data' folder.


## Training

1. **Write Config File**

We need config file to start training. The example config file consists option like below.

```
train:
    name: EFGC  # Experiment result folder name
    batch_size: 64  # Batch size for a each task
    checkpoints_dir: ./checkpoints  #  Root result directory
    gpu_ids: [0]    # GPU ids: e.g. [0] [0,1,2]. use [] for CPU
    epoch_count: 1
    n_epochs: 100   # Number of epochs
    save_weight: true   # Do not save model parameter when set to false

dataset:
    data_root: ./data   # Path to root data directory
    rt_data: EMNIST # ToI dataset name [MNIST | FashionMNIST | EMNIST]
    irt_data: FashionMNIST  # IrT dataset name [MNIST | FashionMNIST | EMNIST]
    s_domain: G # Source domain [G | C | E | N]
    t_domain: C # Target domain [G | C | E | N]
    rt_classes: 27  # Number of classes in ToI
    irt_classes: 10 # Number of classes in IrT
    img_size: 28    # Input image size
    workers: 8  # Number of workers for dataloader
    match_sampling: true    # Enable class-matching batch sampling method 

optimizer:
    lr: 0.0002  # Initial learning rate for adam optimizer
    lr_policy: step
    lr_decay_iters: 30  # Multiply by a gamma every lr_decay_iters iterations
    n_epochs_decay: 30  # Number of epochs to linearly decay learning rate to zero
    beta1: 0.5  # Momentum term of adam

logger:
    print_loss: false   # Whether show loss
    print_freq: 500 # Frequency of showing loss

models:
    FS: SA
    FS_n_layers: 2
```

2. **Start Training**

Train the model:

`python train.py --config CONFIG_FILE`


## Citation

```
@inproceedings{Jhoo_2021_ICCV,
    title     = {Collaborative Learning With Disentangled Features for Zero-Shot Domain Adaptation},
    author    = {Jhoo, Won Young and Heo, Jae-Pil},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021}
}
```