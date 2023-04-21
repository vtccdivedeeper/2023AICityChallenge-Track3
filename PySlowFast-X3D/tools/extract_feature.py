# Source from paper: https://openaccess.thecvf.com/content/CVPR2022W/AICity/papers/Tran_An_Effective_Temporal_Localization_Method_With_Multi-View_3D_Action_Recognition_CVPRW_2022_paper.pdf

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
import glob
from multiprocessing import Process, Queue, Manager, Lock
from slowfast.utils.parallel_infer import extract_frames, extract_worker, predict_worker
import gc

FPS = 30

def imresize(im, dsize):
    '''
    Resize the image to the specified square sizes and 
    maintain the original aspect ratio using padding.
    Args:
        im -- input image.
        dsize -- output sizes, can be an integer or a tuple.
    Returns:
        resized image.
    '''
    if type(dsize) is int:
        dsize = (dsize, dsize)
    im_h, im_w, _ = im.shape
    to_w, to_h = dsize
    scale_ratio = min(to_w/im_w, to_h/im_h)
    new_im = cv2.resize(im,(0, 0), 
                        fx=scale_ratio, fy=scale_ratio, 
                        interpolation=cv2.INTER_AREA)
    new_h, new_w, _ = new_im.shape
    padded_im = np.full((to_h, to_w, 3), 128)
    x1 = (to_w-new_w)//2
    x2 = x1 + new_w
    y1 = (to_h-new_h)//2
    y2 = y1 + new_h
    padded_im[y1:y2, x1:x2, :] = new_im 
    # print('padd', padded_im)
    return padded_im

def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x
      
@torch.no_grad()        
def inference_1video(cfg, video_path, models):
    if isinstance(models, list) is False:
        models = [models]

    _t_total = time.time()
    print(f"===== Video path: {video_path}")

    _t = time.time()
    clips = extract_frames(video_path, cfg, use_lib="cv2")   
    num_clips = len(clips)
    print("Time for extracting frames: ", time.time() - _t)

    # create clip and process image
    all_inputs = []
    _input = []
    all_batches = []
    batch = []

    # create clip and process image
    _t = time.time()
    all_batches = []
    batch = []
    cnt_batch = 0
    for clip in clips:
        # batching
        batch.append(clip)
        cnt_batch += 1
        if cnt_batch == cfg.TEST.BATCH_SIZE:
            batch = torch.stack(batch, axis=0).cuda(non_blocking=True)
            all_batches.append([batch])
            batch = []
            cnt_batch = 0
    
    if len(batch) > 0: # the last batch hasn't enough datapoints
        # Sample frames for the fast pathway.
        batch = torch.stack(batch, axis=0).cuda(non_blocking=True)
        all_batches.append([batch])

    print("Time for loading data:", time.time() - _t)
    print("Example the first batch shape:", all_batches[0][0].shape)
    print("Example the last batch shape:", all_batches[-1][0].shape)
    print("Number of batches:", len(all_batches))


    predict_sq = []
    prob_sq = []
    score_sq = []
    print('')
    # inference
    for batch in all_batches:
        preds_list = [model(batch).detach().cpu().numpy() for model in models]  
        # print('preds_list.shape', preds_list[0].shape)
        prob_ensemble = np.array(preds_list)

        if dataset_name == "SetA1":
            prob_ensemble = np.mean(prob_ensemble, axis=0) # Batch_size x 2048
        elif dataset_name == "SetA2":
            if use_ensemble_fold:
                prob_ensemble = np.mean(prob_ensemble, axis=0) # Batch_size x 2048
            # else do nothing: # 5 x Batch_size x 2048
        prob_sq.append(prob_ensemble)

    del all_batches
    gc.collect()
    if dataset_name == "SetA1":
        feature = np.concatenate(prob_sq, axis=0) # Duration x 2048
    elif dataset_name == "SetA2":
        if use_ensemble_fold:
            feature = np.concatenate(prob_sq, axis=0) # Duration x 2048
        else:
            feature = np.concatenate(prob_sq, axis=1) # 5 x Duration x 2048
    
    print("Total time: ", time.time() - _t_total)
    return feature

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

@torch.no_grad()   
def load_model(cfg, checkpoint):
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()   
    return model

NUM_FOLD = 5
dataset_name = "SetA2" 
use_ensemble_fold = True

if __name__ == "__main__":  
    # LOAD CONFIG 
    args = parse_args()
    cfg = load_config(args, args.cfg_files[0])
    cfg = assert_and_infer_cfg(cfg)
    seed_everything(719)

    # CREATE OUTPUT DIR
    output_root_dir = cfg.DATA.PATH_EXTRACT
    print('output_root_dir', output_root_dir)
    if os.path.exists(output_root_dir) == False:
        os.mkdir(output_root_dir)

    # PRINT INFORMATION
    labels = [i for i in range(cfg.MODEL.NUM_CLASSES)]
    path = cfg.DATA.PATH_TO_DATA_DIR
    print("="*20)
    print("INPUT PATH:", path)
    print("OUTPUT PATH:", cfg.DATA.PATH_EXTRACT)
    print(f"NUM_FRAMES: {cfg.DATA.NUM_FRAMES} | SAMPLING_RATE: {cfg.DATA.SAMPLING_RATE} | TEST BATCH_SIZE: {cfg.TEST.BATCH_SIZE}")

    print("======= INFERENCE =========")
    
    # views = ["right"]
    # checkpoint_lists = [cfg.WEIGHT.RIGHT_SIDE_WINDOW]
    views = ["dash", "rear", "right"]
    checkpoint_lists = [cfg.WEIGHT.DASHBOARD, cfg.WEIGHT.REARVIEW, cfg.WEIGHT.RIGHT_SIDE_WINDOW]
    
    column_idx_csv = [1,2,3]
    preds = {}
    input_user_paths = sorted(glob.glob(os.path.join(cfg.DATA.PATH_TO_DATA_DIR, "user_id_*")))
    # print(input_user_paths)

    # view -> user:
    # case 1 (for setA2): 
        # case 1.1: use_ensemble_fold: average of feature in 5 folds
        # case 1.2: seperate feature for 5 fold
    # case 2 (for setA1): 
        # case 1.1: use_ensemble_fold: average of feature in 5 folds
        # case 2.1: 1 fold ~ 5 valid user
    for idx_view, view in enumerate(views):
        for idx_user, input_user_path in enumerate(input_user_paths):
            user_name = input_user_path.split("/")[-1]
            print('input_user_path', input_user_path)

            if dataset_name == "SetA2" and use_ensemble_fold == False:
                for idx_fold in range(NUM_FOLD):
                    output_fold_path = os.path.join(output_root_dir, f"fold{idx_fold}")
                    os.makedirs(output_fold_path, exist_ok=True)

                    output_user_path = os.path.join(output_fold_path, user_name)
                    print('output_user_path', output_user_path)
                    if os.path.exists(output_user_path) == False:
                        os.mkdir(output_user_path)
            else:
                output_user_path = os.path.join(output_root_dir, user_name)
                print('output_user_path', output_user_path)
                if os.path.exists(output_user_path) == False:
                    os.mkdir(output_user_path)

            if dataset_name == "SetA2":
                # print("Use ensemble fold")
                print(f"Checkpoints: {checkpoint_lists[idx_view]}")         
                model = [load_model(cfg, checkpoint_lists[idx_view][idx_fold]) for idx_fold in range(NUM_FOLD)]

                input_user_video_paths = glob.glob(os.path.join(input_user_path, "*.MP4"))
                for input_user_video_path in input_user_video_paths:
                    video_name = input_user_video_path.split("/")[-1]
                    if view in input_user_video_path.lower():
                        print('input_user_video_path', input_user_video_path) # A/B/C/userX/view.MP4
    
                        feature = inference_1video(cfg, input_user_video_path, model)
                        print('feature.shape', feature.shape)
                        
                        if len(feature.shape) == 3:
                            for idx_fold in range(feature.shape[0]):
                                # # input_data_dir/userX/view.MP4 -> output_data_dir/fold_i/userX/view.MP4
                                output_user_feature_path =  os.path.join(output_root_dir, f"fold{idx_fold}/{user_name}/{video_name}.npy")
                                np.save(output_user_feature_path, feature[idx_fold])
                                print(f"=== Write to file: {output_user_feature_path} ====")   
                        else: 
                            output_user_feature_path =  os.path.join(output_root_dir, f"{user_name}/{video_name}.npy")
                            np.save(output_user_feature_path, feature)
                            print(f"=== Write to file: {output_user_feature_path} ====")   

            elif dataset_name == "SetA1":
                if use_ensemble_fold:
                    # print("Use ensemble fold")
                    print(f"Checkpoints: {checkpoint_lists[idx_view]}")         
                    model = [load_model(cfg, checkpoint_lists[idx_view][idx_fold]) for idx_fold in range(NUM_FOLD)]

                    input_user_video_paths = glob.glob(os.path.join(input_user_path, "*.MP4"))
                    for input_user_video_path in input_user_video_paths:
                        video_name = input_user_video_path.split("/")[-1]
                        if view in input_user_video_path.lower():
                            print('input_user_video_path', input_user_video_path) # A/B/C/userX/view.MP4
        
                            feature = inference_1video(cfg, input_user_video_path, model)
                            print('feature.shape', feature.shape)
                            
                            if len(feature.shape) == 3:
                                for idx_fold in range(feature.shape[0]):
                                    # # input_data_dir/userX/view.MP4 -> output_data_dir/fold_i/userX/view.MP4
                                    output_user_feature_path =  os.path.join(output_root_dir, f"fold{idx_fold}/{user_name}/{video_name}.npy")
                                    np.save(output_user_feature_path, feature[idx_fold])
                                    print(f"=== Write to file: {output_user_feature_path} ====")   
                            else: 
                                output_user_feature_path =  os.path.join(output_root_dir, f"{user_name}/{video_name}.npy")
                                np.save(output_user_feature_path, feature)
                                print(f"=== Write to file: {output_user_feature_path} ====")                
                else:
                    for idx_fold in range(NUM_FOLD):
                        if idx_fold == idx_user % NUM_FOLD:
                            print('CHECK_POINT', checkpoint_lists[idx_view][idx_fold])
                            model = load_model(cfg, checkpoint_lists[idx_view][idx_fold])
                            input_user_video_paths = glob.glob(os.path.join(input_user_path, "*.MP4"))
                            for input_user_video_path in input_user_video_paths:
                                video_name = input_user_video_path.split("/")[-1]
                                if view in input_user_video_path.lower():
                                    print('input_user_video_path', input_user_video_path)

                                    feature = inference_1video(cfg, input_user_video_path, model)
                                    print('feature.shape', feature.shape)

                                    if len(feature.shape) == 3:
                                        for idx_fold in range(feature.shape[0]):
                                            # # input_data_dir/userX/view.MP4 -> output_data_dir/fold_i/userX/view.MP4
                                            output_user_feature_path =  os.path.join(output_root_dir, f"fold{idx_fold}/{user_name}/{video_name}.npy")
                                            np.save(output_user_feature_path, feature[idx_fold])
                                            print(f"=== Write to file: {output_user_feature_path} ====")   
                                    else: 
                                        output_user_feature_path =  os.path.join(output_root_dir, f"{user_name}/{video_name}.npy")
                                        np.save(output_user_feature_path, feature)
                                        print(f"=== Write to file: {output_user_feature_path} ===") 
            else:
                print("Invalid dataset name")
                raise Exception