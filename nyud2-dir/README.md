# ConR on NYUD2-DIR
This repository contains the implementation of __ConR__ on NYUD2-DIR

The imbalanced regression framework and LDS+FDS are based on the public repository of [Ren et al., CVPR 2022](https://github.com/jiawei-ren/BalancedMSE). 

## Installation

#### Prerequisites

1. Download and extract NYU v2 dataset to folder `./data` using

```bash
python download_nyud2.py
```

2. __(Optional)__ We use required meta files `nyu2_train_FDS_subset.csv` and `test_balanced_mask.npy`  provided by Yang et al.(ICML 2021), which is used to set up  efficient FDS feature statistics computation and balanced test set mask in folder `./data`. To reproduce the results in the paper, please directly use these two files. For different FDS computation subsets and balanced test set masks, you can run

```bash
python preprocess_nyud2.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- numpy, pandas, scipy, tqdm, matplotlib, PIL, gdown, tensorboardX


## Getting Started

### 1. Train baselines

To use Balanced MSE

```bash
python train.py --bmse --imp bni --init_noise_sigma 1.0 --fix_noise_sigma
```



### 2. Train a model with ConR



```bash
python train.py --conr -w 0.2 --beta 0.2 -e 0.2
```
### 3. Evaluate and reproduce


```bash
python test.py --eval_model <path_to_evaluation_ckpt>
```