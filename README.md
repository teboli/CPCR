# CPCR

Code for the paper "End-to-end Interpretable Learning of Non-blind Image Deblurring" at ECCV2020 by Jian Sun, Jean Ponce and I.

### Installation

Please first run the setup.py file with `python setup.py install` to compile the custom non-uniform convolution layer. This should work for any GPU.

### Training

We provide the training code for both uniform and non-uniform setting in `train_lchqs.py` and `train_nulchqs.ps` where the default parameters enables to reproduce the results in the paper:

* Uniform training (blind setting): `python train_lchqs.py --n_epochs 150 --n_in 2 --n_out 5`;
* Uniform training (non-blind setting): `python train_lchqs.py --n_epochs 150 --n_in 2 --n_out 5 --blind 0`;
* Non-uniform training (blind setting): `python train_nulchqs.py --n_epochs 150 --n_in 2 --n_out 5`;
* Non-uniform training (non-blind setting): `python train_nulchqs.py --n_epochs 150 --n_in 2 --n_out 5 --blind 0`;

### Evaluation

One can also test the (pre)trained models on the 640 images from [Sun et al., Edge-based blur kernel estimation using patch priors, ICIP13], the uniform and non-uniform benchmarks of 160 and 100 images taken from PASCAL VOC we introduced, the realistic images shipped with the code of [Pan et al., Blind Image Deblurring Using Dark Channel Prior, CVPR16] with pure Python code and on the 32 images from [Levin et al., Understanding and evaluating blind deconvolution algorithms, CVPR09] with a Bash script that runs Python and Matlab code (evaluating Anat Levin's requires running `comp_upto_shift.m`):

* Evaluate on Levin dataset: `bash evaluate_lchqs_levin.sh --n_load 300 --n_in --n_out 5`;
* Evaluate on Sun dataset: `python evaluate_lchqs.py --n_load 300 --n_in 2 --n_out 5 --dataset sun`;
* Evaluate on uniform Pascal dataset: `python evaluate_lchqs.py --n_load 300 --n_in 2 --n_out 5 --dataset pascal --pascal_noise 3`;
* Evaluate on non-uniform Pascal dataset: `python evaluate_nulchqs.py --n_load 300 --n_in 2 --n_out 5 --dataset pascal --pascal_noise 3`;
* Evaluate on realistic dataset: `python evaluate_lchqs.py --n_load 300 --n_in 2 --n_out 5 --dataset realistic`;

### Troubleshouting

In case of any trouble, please reach me at `thomas.eboli@inria.fr` or open an issue!
