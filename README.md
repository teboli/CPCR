# CPCR

This repository is for the non-blind deblurring method introduced in the following paper

[Thomas Eboli](https://www.di.ens.fr/thomas.eboli/), [Jian Sun](http://gr.xjtu.edu.cn/web/jiansun) and [Jean Ponce](https://www.di.ens.fr/~ponce/), "End-to-end Interpretable Learning of Non-blind Image Deblurring", ECCV 2020, [[arXiv]](https://arxiv.org/abs/2007.01769).

### Installation

Please first run the setup.py file with `python setup.py install` to compile the custom non-uniform convolution layer. This should work for any GPU.

### Training

We provide the training code for both uniform and non-uniform setting in `train_lchqs.py` and `train_nulchqs.ps` where the default parameters enables to reproduce the results in the paper:

* Uniform training (blind setting): `python train_lchqs.py --n_epochs 150 --n_in 2 --n_out 5 --sigma 5.0`;
* Uniform training (non-blind setting): `python train_lchqs.py --n_epochs 150 --n_in 2 --n_out 5 --blind 0 --sigma 1.0`;
* Non-uniform training (blind setting): `python train_nulchqs.py --n_epochs 150 --n_in 2 --n_out 5 --sigma 5.0`;
* Non-uniform training (non-blind setting): `python train_nulchqs.py --n_epochs 150 --n_in 2 --n_out 5 --blind 0 --sigma 1.0`;

First line is blind training of the uniform model with Gaussian noises between 0.5 and 12.75 (5%) of magnitude. Second line is non-blind training of the uniform model with Gaussian noises 2.55 (1%) of magnitude. The third and fourth lines are the equivalent training scenarios for non-uniform deblurring.

### Evaluation

One can directly use our pretrained model. We provide one example of synthetic blurry and noisy (3% Gaussian noise) with uniform blur, one example of synthetic blurry and noisy (1% Gaussian noise) with non-uniform motion blur obtained with the code of [Gong et al., From Motion Blur to Motion Flow: a Deep Learning Solution for Removing Heterogeneous Motion Blur, CVPR 2017] and three real-world blurry images with kernels estimated with the code of [Pan et al., Blind Image Deblurring Using Dark Channel Prior, CVPR 2016].

### Troubleshouting

In case of any trouble, please reach me at `thomas.eboli@inria.fr` or open an issue.

### Citation

If you find the code helpful in your resarch or work, please cite the following paper.
```
@inproceedings{eboli2020end2end,
    title={End-to-end Interpretable Learning of Non-blind Image Deblurring},
    author={Eboli, Thomas and Sun, Jian and Ponce, Jean},
    booktitle={ECCV},
    year={2020}
}
```
