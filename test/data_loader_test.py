#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compute model and loader timings."""

import pytorch_cls.core.benchmark as benchmark
import pytorch_cls.core.config as config
import pytorch_cls.core.logging as logging
import pytorch_cls.datasets.loader as loader
from pytorch_cls.core.config import cfg

logger = logging.get_logger(__name__)


def main():
    config.load_cfg_fom_args("Compute model and loader timings.")
    # config.assert_and_infer_cfg()
    test_loader = loader.construct_test_loader()
    logging.setup_logging()
    avg_time = benchmark.compute_full_loader(test_loader, epoch=2)
    for i, _time in enumerate(avg_time):
        logger.info("The {}'s epoch average time is: {}".format(i, avg_time))

if __name__ == "__main__":
    main()
