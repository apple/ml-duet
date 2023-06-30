# DUET: 2D Structured and Aproximately Equivariant Representations

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This software project accompanies the research paper, [DUET: 2D Structured and Approximately Equivariant Representations, ICML 2023](https://arxiv.org/abs/2306.16058).

### Abstract

Multiview Self-Supervised Learning (MSSL) is based on learning invariances with respect to a set of input transformations. However, invariance partially or totally removes transformation-related information from the representations, which might harm performance for specific downstream tasks that require such information. We propose 2D strUctured and approximately EquivarianT representations (coined DUET), which are 2d representations organized in a matrix structure, and equivariant with respect to transformations acting on the input data. DUET representations maintain information about an input transformation, while remaining semantically expressive. Compared to SimCLR (Chen et al., 2020) (unstructured and invariant) and ESSL (Dangovski et al., 2022) (unstructured and equivariant), the structured and equivariant nature of DUET representations enables controlled generation with lower reconstruction error, while controllability is not possible with SimCLR or ESSL. DUET also achieves higher accuracy for several discriminative tasks, and improves transfer learning.

## Documentation

The requirements are listed in [frozen_requirements.txt](frozen_requirements.txt). The code has been tested using `Python 3.8.10` on MacOS and Linux Ubuntu 18.04. Run the following for installation:


#### Create a virtual environment
```bash
cd <path_to_this_project>
python3 -m venv env  # make sure Python3 >= 3.8
source env/bin/activate
```

#### Install duet dependencies
```bash
pip install -r frozen_requirements.txt
pip install -e .
```

## Getting Started 

This repository contains the main code to reproduce the results in Figure 4 and Tables 2, 3 of the paper for CIFAR10/100. One can also train with MNIST.

> **Note on distributed training:** To democratize the code, we implementation provided runs on a single NVIDIA A100 GPU (80Gb) at batch size 1024. 
The results might slightly differ from those in the paper (batch size 2048), 
we leave the implementation of distributed training out of the scope of this project. IF you have a smaller GPU consider reducing the batch size, or using a smaller `--encoder`.

The main script of interest is [scripts/train_ssl.py](scripts/train_ssl.py). Usage:

```shell
usage: train_ssl.py [-h] [--device {cpu,cuda}] [--dataset STR] [--dataset-root PATH] [--batch-size INT] [--image-size INT] [--center-crop] [--save-dir PATH] [--wandb-project STR] [--model {duet,simclr,essl}]
                    [--encoder {Cifar10_ResNet,cifar10_resnet20,cifar10_resnet32,cifar10_resnet44,cifar10_resnet56,cifar10_resnet110,cifar10_resnet1202,lifted_cifar10_resnet20,lifted_cifar10_resnet32,lifted_cifar10_resnet44,lifted_cifar10_resnet56,lifted_cifar10_resnet110,lifted_cifar10_resnet1202}]
                    [--num-classes INT] [--epochs INT] [--warmup-epochs INT] [--test-every INT] [--lr FLOAT] --tx [{sca,rot,ron,fli,flv,blu,gra,jit,res,ccr} [{sca,rot,ron,fli,flv,blu,gra,jit,res,ccr} ...]] --equi-tx {sca,rot,ron,fli,flv,blu,gra,jit,res,ccr} [--duet-lambda FLOAT] [--duet-bins INT]
                    [--duet-target {ga,vm}] [--essl-lambda FLOAT] [--debug-steps INT]

optional arguments:
  -h, --help            show this help message and exit
  --device {cpu,cuda}   The device, wither cpu or cuda. If None, will use cuda when available.
  --dataset STR         The dataset to use.
  --dataset-root PATH   Where to save the dataset locally.
  --batch-size INT      Training batch size.
  --image-size INT      Input image size. All images will be resized to a square of such size.
  --center-crop         If True, images are cropped at their center, omiting boundaries. Default False.
  --save-dir PATH       Where to save the final model.
  --wandb-project STR   Optionally, log to WandB. Requires a file .wandb.yaml with credentials (see README.md).
  --model {duet,simclr,essl}
                        Model type among duet, simclr or essl.
  --encoder {Cifar10_ResNet,cifar10_resnet20,cifar10_resnet32,cifar10_resnet44,cifar10_resnet56,cifar10_resnet110,cifar10_resnet1202,lifted_cifar10_resnet20,lifted_cifar10_resnet32,lifted_cifar10_resnet44,lifted_cifar10_resnet56,lifted_cifar10_resnet110,lifted_cifar10_resnet1202}
                        Encoder backbone architecture.
  --num-classes INT     Number of classes in the training set.
  --epochs INT          Training epochs.
  --warmup-epochs INT   Learning rate linear warm up in epochs.
  --test-every INT      Every how many epochs should the test dataset be evaluated.
  --lr FLOAT            The train learning rate.
  --tx [{sca,rot,ron,fli,flv,blu,gra,jit,res,ccr} [{sca,rot,ron,fli,flv,blu,gra,jit,res,ccr} ...]]
                        Image transformations to be applied, in order of application.
  --equi-tx {sca,rot,ron,fli,flv,blu,gra,jit,res,ccr}
                        That transformation we want to learn its structure (or we want to become equivariant to). Must be one of `--tx` for DUET and one not in `--tx` for ESSL.
  --duet-lambda FLOAT   Lambda parameter in DUET, weighs the structure loss.
  --duet-bins INT       Number of bins in the structure axis of DUET, parameter G in the paper.
  --duet-target {ga,vm}
                        Target distribution, either `ga` (Gaussian) or `vm` (von-Mises). The parameter \sigma in the paper will be automatically chosen accordingly.
  --essl-lambda FLOAT   Lambda parameter in ESSL, weighs the equivariance loss. Defaults to 0.4 according to the ESSL paper.
  --debug-steps INT     Number of steps per epoch, use 2 or 3 to debug and not have to wait for a full epoch.
```

For example, to train DUET on CIFAR10 with RandomResizedCrop + RandomRotation, and learn structure wrt. Rotation, we would call:
```shell
python scripts/train_ssl.py --model duet --encoder cifar10_resnet32 --dataset cifar10 --batch-size 1024 --test-every 10 --warmup-epochs 10 --epochs 800 --center-crop --tx sca rot --equi-tx rot --duet-target vm --duet-lambda 1000 --save-dir some_dir
```

For such a model, we'd expect an accuracy of ~73%, as in Figure 4 in the paper.

## Logging results

You can log your results using [WandB](https://wandb.ai/site). First install `pip install -U wandb` and create a file called `.wandb.yaml` in the project root directory with the following content:

```yaml
WANDB_API_KEY: <YOUR_API_KEY>
WANDB_BASE_URL: <YOUR_WANDB_URL>
```

Then, just use the `--wandb-project YOUR_PROJECT_NAME` argument when running `train_ssl.py` and the logs will be redirected to WandB.

## Citation

```bibtex
@article{suau2023duet,
  title={DUET: 2D Structured and Approximately Equivariant Representations},
  author={Suau, Xavier and Danieli, Federico and Keller, Anderson T. and Blaas, Arno and Huang, Chen and Ramapuram, Jason and Busbridge, Dan and Zappella, Luca},
  journal={International Conference on Machine Learning},
  year={2023}
}
```

## Contact

Xavier Suau Cuadros (`xsuaucuadros@apple.com`)

