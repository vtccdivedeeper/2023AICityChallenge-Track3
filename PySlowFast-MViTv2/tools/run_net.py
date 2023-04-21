#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize
import os


VIEWS = ["Dashboard", "Right_side_window", "Rearview"]
FOLDS = [0, 1, 2, 3, 4]
CHECKPOINTS = {"Dashboard": ["checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00107.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00105.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00192.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00046.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00060.pyth"], 
                "Right_side_window": ["checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00197.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00107.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00180.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00049.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00133.pyth"], 
                "Rearview": ["checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00197.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00116.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00101.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00060.pyth",
                            "checkpoint/18_03/fold{}/{}/checkpoints/checkpoint_epoch_00175.pyth"]}

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

        data_path_original = cfg.DATA.PATH_TO_DATA_DIR
        print("ORIGINAL: ", cfg.DATA)
        output_path_original = cfg.OUTPUT_DIR

        for fold in FOLDS:
            for view in VIEWS:
                print("FOLD: ", fold)
                cfg.DATA.PATH_TO_DATA_DIR = os.path.join(data_path_original, f"fold{fold}/{view}")
                cfg.OUTPUT_DIR = os.path.join(output_path_original, f"fold{fold}/{view}")
                cfg.TEST.CHECKPOINT_FILE_PATH = CHECKPOINTS[view][fold].format(fold, view)
                print(cfg.DATA.PATH_TO_DATA_DIR)
                print(cfg.OUTPUT_DIR)
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                # Perform training.
                if cfg.TRAIN.ENABLE:
                    launch_job(cfg=cfg, init_method=args.init_method, func=train)
                # Perform multi-clip testing.
                if cfg.TEST.ENABLE:
                    if cfg.TEST.NUM_ENSEMBLE_VIEWS == -1:
                        num_view_list = [1, 3, 5, 7, 10]
                        for num_view in num_view_list:
                            cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view
                            launch_job(cfg=cfg, init_method=args.init_method, func=test)
                    else:
                        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
