# Model Zoo (Copyed and modified from [pycls](https://github.com/facebookresearch/pycls/edit/master/MODEL_ZOO.md))

## Introduction

This file documents a collection of baselines trained with **pytorch-cls**. All configurations for these baselines are located in the `configs/baselines` directory. The tables below provide results and useful statistics about training and inference. Links to the pretrained models are provided as well. The following experimental and training settings are used for all of the training and inference runs.

### Experimental Settings

- All baselines were run on 8 NVIDIA Tesla V100 GPUs (32GB GPU memory).
- All baselines were run using PyTorch 1.5, CUDA 10.1, and cuDNN 7.0.
- Inference times are reported for 64 images on 1 GPU for all models.
- Training times are reported for 100 epochs on 8 GPUs with the batch size listed.
- Due to the computation limitation, we only run 1 time to generate the reported errors.
- All models and results below are on the ImageNet-1k dataset.

### Training Settings

Our primary goal is to provide simple and strong baselines that are easy to reproduce. For all models, we use our basic training settings without any training enhancements (e.g., DropOut, DropConnect, AutoAugment, EMA, etc.) or testing enhancements (e.g., multi-crop, multi-scale, flipping, etc.); (Please note that, networks in [pycls](https://github.com/facebookresearch/pycls/edit/master/MODEL_ZOO.md) are trained with an extra PCA lighting, we delete this data augmentation for a eaiser reproduce.)

- We use SGD with momentum of 0.9, a half-period cosine schedule, and train for 100 epochs.
- For ResNet/ResNeXt/RegNet, we use a *reference* learning rate of 0.1 and a weight decay of 5e-5.
- For EfficientNet, we use a *reference* learning rate of 0.2 and a weight decay of 1e-5.
- The actual learning rate for each model is computed as (batch-size / 128) * reference-lr.
- For training, we use basic transformations including aspect ratio, flipping and per-channel mean and SD normalization.
- At test time, we rescale images to (256 / 224) * train-res and take the center crop of train-res.
- For ResNet/ResNeXt/RegNet, we use the image size of 224x224 for training.
- For EfficientNet, the training image size varies following the original paper.

### DataLoader

We provide imagenet dataloader with different backends

- Custom: The image is read with opencv, all the transformers is copyed from [pycls](https://github.com/facebookresearch/pycls/edit/master/MODEL_ZOO.md)

- DALI_CPU: Using dali for image decoding and tranformation, all the operations is running on CPU.

- DALI_GPU: Using dali for image decoding and tranformation, all the operations is running on GPU.

- Torch: Using torch as dataloader, all the transformer are from torchvision.

All the baselines are trained with DALI_CPU backend

## Baselines

### ResNet

| Model    |FLOPS(B)|params(M)|acts(M)|batch size|infer(ms)|train(hrs)|Top1 |download|
| -------  |:----:  |:-------:|:-----:|:--------:|:-------:|:-------: |:---:|:---:   |
| ResNet-50|4.1     |22.6     |11.1   |256       |53       |22.5      |23.46| -      |
|ResNet-101|7.8     |44.6     |16.2   |256       |90       |35.4      |21.60| -      |
|ResNet-152|11.5    |60.1     |22.5   |256       |53       |38.75     |21.08| -      |

### Efficient

| Model         |FLOPS(B)|params(M)|acts(M)|batch size|infer(ms)|train(hrs)|Top1 |download|
| -------       |:----:  |:-------:|:-----:|:--------:|:-------:|:-------: |:---:|:---:   |
|EfficientNet-B0|0.4     |5.3      |6.7    |256       |34       |13.64     |25.37| -      |
|EfficientNet-B1|0.7     |7.8      |10.9   |256       |52       |25.65     |24.25| -      |
