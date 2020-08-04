#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test build a classification model."""

import torch
from thop import profile as thop_profile

import pytorch_cls.core.config as config
from pytorch_cls.core.builders import build_model
from pytorch_cls.core.config import cfg
from pytorch_cls.core.net import complexity
from pytorch_cls.core.complexity_counter import profile


def main():
    config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    print("building model {}".format(cfg.MODEL.TYPE))
    model = build_model()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    model_complex = complexity(model)
    print(model_complex)
    model_complex_2 = profile(model, [1, 3, 224, 224])
    print(model_complex_2)
    a, b = thop_profile(model, [1, 3, 224, 224])
    print(a)
    print(b)


if __name__ == "__main__":
    main()
