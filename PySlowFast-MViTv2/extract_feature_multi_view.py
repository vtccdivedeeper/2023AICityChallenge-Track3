
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import os
import sys
import pickle
import torch
import random
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
import cv2
import pandas as pd
import tqdm
"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import time
logger = logging.get_logger(__name__)
import csv
from itertools import islice


@torch.no_grad()        
def extract_fold(checkpoint_path, cfg, val_df):
    """
    Main function to spawn the train and test process.
    """
    
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)
    # print(cfg, checkpoint_path)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_path
    cfg.TRAIN.ENABLE = "False"
    cfg.NUM_GPUS = 1
    
    print('cfg', cfg, checkpoint_path)
    du.init_distributed_training(cfg)
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    

    total_prob_sq={}
    video_order = []
    for index, row in val_df.iterrows():
        print(row['Video_path'], row['Label'])
    #     for able_to_read, frame in img_provider:
    #         count += 1
    #         i+=1
    #         if not able_to_read:
    #             # when reaches the end frame, clear the buffer and continue to the next one.
    #             frames = []
    #             # continue
    #             break
    #         if len(frames) != cfg.DATA.NUM_FRAMES and count % cfg.DATA.SAMPLING_RATE ==0:
    #             # frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frame_processed = cv2.resize(frame_processed, (224, 224))
    #             frames.append(frame_processed)
    #         if len(frames) == cfg.DATA.NUM_FRAMES:
    #             start = time.time()
    #             # Perform color normalization.
    #             inputs = torch.tensor(np.array(frames)).float()
    #             inputs = inputs / 255.0
    #             inputs = inputs - torch.tensor(cfg.DATA.MEAN)
    #             inputs = inputs / torch.tensor(cfg.DATA.STD)
    #             # print(cfg.DATA.MEAN, cfg.DATA.STD)
    #             # 
    #             # T H W C -> C T H W.
    #             inputs = inputs.permute(3, 0, 1, 2)
    #             # 1 C T H W.
    #             inputs = inputs[None, :, :, :, :]
    #             # Sample frames for the fast pathway.
    #             index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
    #             fast_pathway = torch.index_select(inputs, 2, index)
    #             inputs = [inputs]
    #             # Transfer the data to the current GPU device.
    #             if isinstance(inputs, (list,)):
    #                 for i in range(len(inputs)):
    #                     inputs[i] = inputs[i].cuda(non_blocking=True)
    #             else:
    #                 inputs = inputs.cuda(non_blocking=True)
    #             # print('inputs[0].shape', inputs[0].shape)
    #             # Perform the forward pass.
    #             preds  = model(inputs).detach().cpu().numpy()   
    #             preds_2  = model_2(inputs).detach().cpu().numpy()   
    #             preds_3  = model_3(inputs).detach().cpu().numpy()   
    #             preds_4  = model_4(inputs).detach().cpu().numpy()   
    #             preds_5  = model_5(inputs).detach().cpu().numpy()   
    #             prob_ensemble = np.array([preds, preds_2, preds_3, preds_4, preds_5])
    #             prob_ensemble = np.mean(prob_ensemble, axis=0)
    #             prob_sq.append(prob_ensemble)
    #             frames = []
    #     total_prob_sq[values[0]] = prob_sq
    # return dict(sorted(total_prob_sq.items())), video_order


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    seed_everything(719)
    args = parse_args()
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
        
        checkpoint_list = [{"Dashboard": './checkpoint/13_03/fold0/Dashboard/checkpoints/checkpoint_epoch_00101.pyth', 
                        "Rearview": './checkpoint/11_03/fold0/Rearview/checkpoints/checkpoint_epoch_00003.pyth',
                        "Right_side_window": './checkpoint/11_03/fold0/Right_side_window/checkpoints/checkpoint_epoch_00003.pyth'},
                        {"Dashboard": './checkpoint/11_03/fold1/Dashboard/checkpoints/checkpoint_epoch_00003.pyth',
                        "Rearview": './checkpoint/11_03/fold1/Rearview/checkpoints/checkpoint_epoch_00003.pyth',
                        "Right_side_window": './checkpoint/11_03/fold1/Right_side_window/checkpoints/checkpoint_epoch_00003.pyth'},
                        {"Dashboard": './checkpoint/11_03/fold2/Dashboard/checkpoints/checkpoint_epoch_00003.pyth',
                        "Rearview": './checkpoint/11_03/fold2/Rearview/checkpoints/checkpoint_epoch_00003.pyth',
                        "Right_side_window": './checkpoint/11_03/fold2/Right_side_window/checkpoints/checkpoint_epoch_00003.pyth'},
                        {"Dashboard": './checkpoint/11_03/fold3/Dashboard/checkpoints/checkpoint_epoch_00003.pyth',
                        "Rearview": './checkpoint/11_03/fold3/Rearview/checkpoints/checkpoint_epoch_00003.pyth',
                        "Right_side_window": './checkpoint/11_03/fold3/Right_side_window/checkpoints/checkpoint_epoch_00003.pyth'},
                        {"Dashboard": './checkpoint/11_03/fold4/Dashboard/checkpoints/checkpoint_epoch_00003.pyth',
                        "Rearview": './checkpoint/11_03/fold4/Rearview/checkpoints/checkpoint_epoch_00003.pyth',
                        "Right_side_window": './checkpoint/11_03/fold4/Right_side_window/checkpoints/checkpoint_epoch_00003.pyth'}]  

        oof_features = []
        for fold in [0, 1, 2, 3, 4]:
            view_features = {}
            min_len = 10000000
            for view in ["Dashboard", "Rearview", " Right_side_window"]:
                val_df =  pd.read_csv("/data/train_val_split_3views_more_cls0/fold{}/{}/val.csv".format(fold, view), usecols=[0,1], names=['Video_path', 'Label'], delimiter=" ")
                features = extract_fold(checkpoint_list[fold][view], cfg, val_df)
                view_features[view] = features
                if len(features) > min_len:
                    min_len = len(features)
            
            for i in range(min_len):
                oof_features.append(np.concatenate([view_features["Dashboard"][i], view_features["Rearview"][i], view_features["Right_side_window"][i]], 0))


