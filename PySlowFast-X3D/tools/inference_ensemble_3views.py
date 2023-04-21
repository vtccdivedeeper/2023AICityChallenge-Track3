# Source from paper: https://openaccess.thecvf.com/content/CVPR2022W/AICity/papers/Tran_An_Effective_Temporal_Localization_Method_With_Multi-View_3D_Action_Recognition_CVPRW_2022_paper.pdf

import numpy as np
import os
import sys
import gc
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

FPS = 30
lambda2 = 4
lambda3 = 2

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
def inference_1view(cfg, video_info, checkpoint_list):
    """
    Main function to spawn the train and test process.
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    models_list = []
    for checkpoint in checkpoint_list:
        cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint
        model = build_model(cfg)
        cu.load_test_checkpoint(cfg, model)
        model.eval()
        models_list.append(model)

    print("Number of clips: ", len(video_info))
    print('List checkpoints: ', checkpoint_list)

    total_prob_sq={}
    for video_id, video_path in video_info.items():
        _t_total = time.time() 
        video_path = video_path.replace("AICity2023/data", "AICity2023/raid_data") # use data from raid storage to accelerate the speed
        print('')
        print(f"===== Video path: {video_path}")

        _t = time.time()
        clips = extract_frames(video_path, cfg, use_lib='cv2')  
        num_clips = len(clips)
        print("Time for extracting frames: ", time.time() - _t)

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

        # inference
        for batch in all_batches:
            preds_list = [model(batch).detach().cpu().numpy() for model in models_list]  
            prob_ensemble = np.array(preds_list)
            prob_ensemble = np.mean(prob_ensemble, axis=0)
            prob_sq.append(prob_ensemble)
        
        del all_batches
        gc.collect()

        total_prob_sq[video_id] = np.concatenate(prob_sq)
        print("Total time: ", time.time() - _t_total)

    return total_prob_sq

def get_classification(sequence_class_prob):
    labels_index = np.argmax(sequence_class_prob, axis=1) #returns list of position of max value in each list.
    probs= np.max(sequence_class_prob, axis=1)  # return list of max value in each  list.
    return labels_index, probs

def smoothing(x, k=3):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y
        
def post_processing(cfg, prob_sq, video_id):
    # probability -> logits
    action_class, action_probs = get_classification(prob_sq)
    threshold = np.mean(action_probs)

    # raw data
    startings = []
    endings = []
    for i in range(len(action_class)):
        startings.append(i)
        endings.append(i+1)
    
    raw_df = pd.DataFrame({"video_id":[video_id]*len(action_class), 
                        "activity_id": action_class, 
                        "probability": action_probs,
                        "start_time": startings,
                        "end_time": endings})
    raw_df['start_time'] = raw_df['start_time'].apply(lambda x: x*float(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE) / FPS).astype('int32')
    raw_df['end_time'] = raw_df['end_time'].apply(lambda x: x*float(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE) / FPS).astype('int32')
    raw_df.sort_values(by=['start_time','activity_id','end_time'], ascending=[True, True, True], inplace=True)

    # remove predictions have probs is smaller than mean of probs
    action_tag = np.zeros(action_class.shape)
    action_tag[action_probs >= threshold] = 1 
    activities_idx = []
    startings = []
    endings = []
    for i in range(len(action_tag)):
        if action_tag[i] ==1:
            activities_idx.append(action_class[i])
            startings.append(i)
            endings.append(i+1)
    
    df = pd.DataFrame({"video_id":[video_id]*len(activities_idx), 
                        "activity_id": activities_idx, 
                        "start_time": startings,
                        "end_time": endings})

    df['start_time'] = df['start_time'].apply(lambda x: x*float(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE) / FPS).astype('int32')
    df['end_time'] = df['end_time'].apply(lambda x: x*float(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE) / FPS).astype('int32')
    df.sort_values(by=['start_time','activity_id','end_time'], ascending=[True, True, True], inplace=True)
    print("After filerting low confidence")
    print(df)

    # Merge class
    merged_df = []
    label_idxs = df.activity_id.unique()
    for label_idx in label_idxs:
        sub_df = df.loc[df.activity_id == label_idx].reset_index(drop=True)
        for i in range(len(sub_df)-1):
            if sub_df.loc[i+1, 'start_time'] - sub_df.loc[i, 'end_time'] <= lambda2:
                sub_df.loc[i+1, 'start_time'] = sub_df.loc[i, 'start_time']
                sub_df.loc[i, 'end_time'] = -1
                sub_df.loc[i, 'start_time'] = -1

        merged_df.append(sub_df)

    merged_df = pd.concat(merged_df, ignore_index=True, axis = 0)
    merged_df = merged_df.loc[merged_df['end_time'] != -1]

    print("After merging class")
    print(merged_df)

    # Filter label smaller than lambda 3
    merged_df = merged_df.loc[merged_df['end_time'] - merged_df['start_time'] >= lambda3]

    return merged_df, raw_df

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    
    args = parse_args()
    # print("ARGS:", args)
    cfg = load_config(args, args.cfg_files[0])
    cfg = assert_and_infer_cfg(cfg)

    seed_everything(719)

    labels = [i for i in range(cfg.MODEL.NUM_CLASSES)]
    path = cfg.DATA.PATH_TO_DATA_DIR
    # print("CFG:", cfg)
    print("="*20)
    print("INPUT PATH:", path)
    print("OUTPUT PATH:", cfg.OUTPUT_DIR)
    print(f"NUM_FRAMES: {cfg.DATA.NUM_FRAMES} | SAMPLING_RATE: {cfg.DATA.SAMPLING_RATE} | TEST BATCH_SIZE: {cfg.TEST.BATCH_SIZE}")

    print("=============== INFERENCE =================")
    views = ["Dashboard", "Rearview", "Right_side_window"]
    checkpoint_lists = [cfg.WEIGHT.DASHBOARD, cfg.WEIGHT.REARVIEW, cfg.WEIGHT.RIGHT_SIDE_WINDOW]
    column_idx_csv = [1,2,3]
    preds = {}
    for i, view in enumerate(views):
        print('')
        print(f"============ VIEW: {view} ================")
        name2id = {}
        all_video_names = []
        with open(os.path.join(path, 'video_ids2.csv')) as csvfile:
            csvReader = csv.reader(csvfile, delimiter=',')
            for idx, row in enumerate(csvReader):
                if idx > 0:
                    name2id[row[column_idx_csv[i]]] = row[0]
                    all_video_names.append(row[column_idx_csv[i]])

        video_info = {}
        for root, dirs, files in os.walk(path):
            for vid_name in files:          # Loop over directories, not files
                if vid_name in all_video_names:  # Only keep ones that match
                    video_info[name2id[vid_name]] = os.path.join(root, vid_name)
        video_info = dict(sorted(video_info.items(), key=lambda x: int(x[0])))
        pred = inference_1view(cfg, video_info, checkpoint_lists[i])
        preds[view] = pred

    print("=============== ENSEMLE 3 VIEW =================")
    # ENSEMLE 3 VIEW
    total_video_dfs = []
    raw_list = []
    for video_id, video_path in video_info.items():
        print('')
        print(f'==== Video ID: {video_id} | Path: {video_path}')
        min_len_video = min(len(preds[views[0]][video_id]),
                            len(preds[views[1]][video_id]),
                            len(preds[views[2]][video_id]))
        prob_ensemble_video = (preds[views[0]][video_id][:min_len_video] + preds[views[1]][video_id][:min_len_video] + preds[views[2]][video_id][:min_len_video]) / 3

        # post processing
        video_df, raw_df = post_processing(cfg, prob_ensemble_video, video_id)
        total_video_dfs.append(video_df)
        raw_list.append(raw_df)
    
    print("=============== EXPORT SUBMISSION =================")
    submission_df = pd.concat(total_video_dfs, ignore_index=True, axis=0)
    total_raw_df = pd.concat(raw_list, ignore_index=True, axis=0)

    submission_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'submission_pl.txt'), sep=" ", header=False, index=False)
    total_raw_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'raw_prediction.csv'), index=False)