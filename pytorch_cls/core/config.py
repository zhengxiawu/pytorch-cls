#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys

from yacs.config import CfgNode as CfgNode

# Global config object
_C = CfgNode()
# Example usage:
#   from core.config import cfg
cfg = _C

# ------------------------------------------------------------------------------------ #
# Model options
# ------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------ #
# Common train/test data loader options
# ------------------------------------------------------------------------------------ #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True

# using which backend as image decoder and transformers: dali_cpu, dali_gpu, torch, and cv2
_C.DATA_LOADER.BACKEND = 'dali_cpu'

# Number of data loader workers per process
_C.DATA_LOADER.WORLD_SIZE = 1

_C.DATA_LOADER.PCA_JITTER = False


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file",
                        help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)


def assert_and_infer_cfg(cache_urls=True):
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Precise BN stats computation not verified for > 1 GPU"
    assert not _C.BN.USE_PRECISE_STATS or _C.NUM_GPUS == 1, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
