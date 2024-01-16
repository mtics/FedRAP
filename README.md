# FedRAP

> This project is the code and the supplementary of "**Federated Recommendation with Additive Personalization**"



## Requirements

1. The code is implemented with `Python >= 3.8` and `torch~=1.13.1+cu117`;
2. Other requirements can be installed by `pip install -r requirements.txt`.

## Quick Start

1. First create two folders: `./logs` and `./results`;

2. Put datasets into the path `[parent_folder]/datasets/`;

3. ``````
   python train.py --alias FedRAP --dataset movielens --data_file ml-100k.dat \
       --mu 1e-3 --l2_regularization 1e-6 --lr_network 1e-4 --lr_args 1e3
   ``````

## Contact

- This project is free for academic usage. You can run it at your own risk.
- For any other purposes, please contact Mr. Zhiwei Li ([lizhw.cs@outlook.com](mailto:lizhw.cs@outlook.com))