# FedRAP

[![Static Badge](https://img.shields.io/badge/ICLR-17446-red?style=plastic&logo=iclr&labelColor=%2386C166&color=grey)](https://iclr.cc/virtual/2024/poster/17446) | [![Static Badge](https://img.shields.io/badge/OpenReview-FedRAP-red?style=plastic&logo=OpenReivew&labelColor=%23FCFAF2&color=grey)](https://openreview.net/forum?id=xkXdE81mOK) | [![Static Badge](https://img.shields.io/badge/arxiv-2301.09109-red?style=plastic&logo=arxiv&logoColor=white&labelColor=%23C73E3A&color=grey)](https://arxiv.org/abs/2301.09109)

> This project is the code and the supplementary of "**Federated Recommendation with Additive Personalization**"

> Notice that <u>FedRAP is highly sensitive to the <font color='red'>**Parameter Combinations**</font>, which may result in significant differences in performance!</u>


![Poster of FedRAP @ ICLR 2024](https://iclr.cc/media/PosterPDFs/ICLR%202024/17446.png)


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

## Citation

If you find this paper useful, please cite it:

```
@inproceedings{
    li2024federated,
    title={Federated Recommendation with Additive Personalization},
    author={Zhiwei Li and Guodong Long and Tianyi Zhou},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=xkXdE81mOK}
}
```

## Contact

- This project is free for academic usage. You can run it at your own risk.
- For any other purposes, please contact Mr. Zhiwei Li ([lizhw.cs@outlook.com](mailto:lizhw.cs@outlook.com))