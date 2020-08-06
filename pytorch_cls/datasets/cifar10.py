#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR10 dataset."""

import os
import pickle

import numpy as np
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as torch_transforms
from torch.utils.data.distributed import DistributedSampler

import pytorch_cls.core.logging as logging
import pytorch_cls.datasets.transforms as transforms
from pytorch_cls.core.config import cfg
from pytorch_cls.datasets.transforms import Cutout

logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [125.3, 123.0, 113.9]
_SD = [63.0, 62.1, 66.7]


def Cifar10(data_path, split, batch_size, shuffle, drop_last):
    if cfg.DATA_LOADER.BACKEND == 'custom':
        dataset = Cifar10_custom(data_path, split)
        # Create a sampler for multi-process training
    elif cfg.DATA_LOADER.BACKEND == 'torch':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        transf = [
            torch_transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            torch_transforms.RandomHorizontalFlip()
        ]
        normalize = [
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(MEAN, STD)
        ]

        train_transform = torch_transforms.Compose(transf + normalize)
        valid_transform = torch_transforms.Compose(normalize)
        if cfg.DATA_LOADER.CUTOUT > 0:
            train_transform.transforms.append(Cutout(cfg.DATA_LOADER.CUTOUT))
        dset_cls = dset.CIFAR10
        if split == 'train':
            dataset = dset_cls(root=data_path, train=True,
                               download=True, transform=train_transform)
        elif split == 'val':
            dataset = dset_cls(root=data_path, train=False,
                               download=True, transform=valid_transform)
        else:
            raise NotImplementedError
    else:
        print("cifar10 only support torch and custom beckend!")

    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


class Cifar10_custom(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(
            data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for cifar".format(
            split)
        logger.info("Constructing CIFAR-10 {}...".format(split))
        self._data_path, self._split = data_path, split
        self._inputs, self._labels = self._load_data()

    def _load_data(self):
        """Loads data into memory."""
        logger.info("{} data path: {}".format(self._split, self._data_path))
        # Compute data batch names
        if self._split == "train":
            batch_names = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            batch_names = ["test_batch"]
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._data_path, batch_name)
            with open(batch_path, "rb") as f:
                data = pickle.load(f, encoding="bytes")
            inputs.append(data[b"data"])
            labels += data[b"labels"]
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)
        inputs = inputs.reshape((-1, 3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE))
        return inputs, labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = transforms.color_norm(im, _MEAN, _SD)
        if self._split == "train":
            im = transforms.horizontal_flip(im=im, p=0.5)
            im = transforms.random_crop(
                im=im, size=cfg.TRAIN.IM_SIZE, pad_size=4)
        return im

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = self._prepare_im(im)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]
