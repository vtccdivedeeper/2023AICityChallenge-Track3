# python imports
import argparse
import os
import glob
import time
from pprint import pprint
import pandas as pd
# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed, batched_nms

################################################################################
def infer_1fold(args, cfg, fold, view_type=None):
    print(f"============= FOLD {fold} =============")

    assert len(cfg['infer_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        elif args.epoch == -1:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_best.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    # pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    # add label_dict in case there are not any annotations in json
    label_dict = {}
    for i in range(cfg['dataset']['num_classes']):
        label_dict[i+1] = i
    cfg['dataset']['label_dict'] = label_dict

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['infer_split'], **cfg['dataset']
    )
    print("Number of videos ", len(val_dataset))

    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # # set up evaluator
    # det_eval, output_file = None, None
    # if not args.saveonly:
    #     val_db_vars = val_dataset.get_attributes()
    #     det_eval = ANETdetection(
    #         val_dataset.json_file,
    #         val_dataset.split[0],
    #         tiou_thresholds = val_db_vars['tiou_thresholds']
    #     )
    # else:
    #     output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    results = infer(
        val_loader,
        model,
        print_freq=args.print_freq,
        view_type=view_type
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return results

def infer(
    val_loader,
    model,
    print_freq = 20,
    view_type = None
):
    model.eval()
    # dict for results (for our evaluation code)
    results = dict()

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # unpack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results_single_vid = dict()

                    # Turn Dashboard_user_id_26223_NoAudio_3.MP4 -> user_id_26223_NoAudio_3.MP4
                    video_id = "user_id" + output[vid_idx]['video_id'].split("user_id")[-1]
                    # print(video_id)
                    results_single_vid['video_id'] = video_id
                    results_single_vid['segments'] = output[vid_idx]['segments']
                    results_single_vid['labels'] = output[vid_idx]['labels']
                    results_single_vid['scores'] = output[vid_idx]['scores']
                    
                    if weight_view is not None:
                        for i in range(len(results_single_vid['scores'])):
                            class_id = int(results_single_vid['labels'][i])
                            results_single_vid['scores'][i] = results_single_vid['scores'][i] * weight_view[view_type][class_id]

                    results[video_id] = results_single_vid

        # printing
        # if (iter_idx != 0) and iter_idx % (print_freq) == 0:
        #     # measure elapsed time (sync all kernels)
        #     torch.cuda.synchronize()
        #     batch_time.update((time.time() - start) / print_freq)
        #     start = time.time()

        #     # print timing
        #     print('Test: [{0:05d}/{1:05d}]\t'
        #           'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
        #           iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    return results

def postprocessing(results, test_cfg):
    # input : dict of dictionary items
    # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
    processed_results = []
    results_vid_list = results.values() if isinstance(results, dict) else results
    for results_per_vid in results_vid_list:
        # unpack the meta info

        # 1: unpack the results and move to CPU
        vidx = results_per_vid['video_id']
        segs = results_per_vid['segments']
        scores = results_per_vid['scores']
        labels = results_per_vid['labels']
        if test_cfg["nms_method"] != 'none':
            # 2: batched nms (only implemented on CPU)
            segs, scores, labels = batched_nms(
                segs, scores, labels,
                test_cfg["iou_threshold"],
                test_cfg["min_score"],
                test_cfg["max_seg_num"],
                use_soft_nms = (test_cfg["nms_method"] == 'soft'),
                multiclass = test_cfg["multiclass_nms"],
                sigma = test_cfg["nms_sigma"],
                voting_thresh = test_cfg["voting_thresh"]
            )
        # 3: remove overlap action
        segs, scores, labels = batched_nms(
            segs, scores, labels,
            test_cfg["iou_threshold"],
            0.1,
            test_cfg["max_seg_num"],
            use_soft_nms = False,
            multiclass = True,
            sigma = test_cfg["nms_sigma"],
            voting_thresh = test_cfg["voting_thresh"]
        )
        # 4: repack the results
        processed_results.append(
            {'video_id' : vidx,
                'segments' : segs,
                'scores'   : scores,
                'labels'   : labels}
        )

    return processed_results

weight_view = None
# weight_view = [[0.31438702, 0.76031334, 0.29175307, 0.42156708, 0.35853607,
#         0.73076839, 0.37238685, 0.10172338, 0.47383638, 0.05518508,
#         0.5025323 , 0.75397202, 0.40586585, 0.95405735, 0.12243936],
#        [0.56030248, 0.95931396, 0.70768027, 0.9503711 , 0.70631805,
#         0.79367958, 0.54880619, 0.638063  , 0.47100909, 0.36783717,
#         0.89632257, 0.79389478, 0.01714376, 0.19649477, 0.78744639],
#        [0.34886973, 0.94321252, 0.53478729, 0.65648668, 0.67395198,
#         0.9942615 , 0.56623311, 0.87983649, 0.2284561 , 0.54943332,
#         0.18394702, 0.41540663, 0.17019604, 0.51332716, 0.82586471]]

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('--num_folds', default=5, type=int)  
    args = parser.parse_args()

    """0. load config"""
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    # save the current config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as fid:
        pprint(args, stream=fid)
        fid.flush()

    # INFERENCE
    results_all = []
    ckpt_root = args.ckpt
    folds = [0, 1, 2, 3, 4]
    for fold in folds:
        if cfg['use_concat_3view']:
            args.ckpt = os.path.join(ckpt_root, f"fold{fold}")
            results_all.append(infer_1fold(args, cfg, fold)) # dict(video_id) = dict(video_id, segments, label, score)
        else:
            for idx_view, view_type in enumerate(["Dashboard", "Rearview", "Right_side_window"]):
                args.ckpt = os.path.join(ckpt_root, f"fold{fold}/{view_type}")
                results_all.append(infer_1fold(args, cfg, fold, idx_view)) # dict(video_id) = dict(video_id, segments, label, score)

    # merge all fold and view
    results_final = dict()
    # for fold in range(len(results_all)):
    for fold_idx in range(len(folds)):
        result_fold = results_all[fold_idx]
        fold = folds[fold_idx]
        for video_id in result_fold.keys():
            if video_id not in results_final.keys():
                results_final[video_id] = {"video_id": video_id,
                                        "segments": [],
                                        "scores": [],
                                        "labels": []}
            results_final[video_id]['segments'].append(result_fold[video_id]['segments'])
            results_final[video_id]['scores'].append(result_fold[video_id]['scores'])
            results_final[video_id]['labels'].append(result_fold[video_id]['labels'])

    for video_id in results_final.keys():
        results_final[video_id]['segments'] = torch.cat(results_final[video_id]['segments'])
        results_final[video_id]['scores'] = torch.cat(results_final[video_id]['scores'])
        results_final[video_id]['labels'] = torch.cat(results_final[video_id]['labels'])
    
    # apply nms 
    results_final = postprocessing(results_final, cfg['test_cfg'])

    # convert result into dataframe
    results = {
        'video_id': [],
        'time_start' : [],
        'time_end': [],
        'label': [],
        'score': []
    }
    for result_one_video in results_final:
        if result_one_video['segments'].shape[0] > 0:
            results['video_id'].extend(
                [result_one_video['video_id']] *
                result_one_video['segments'].shape[0]
            )
            results['time_start'].append(result_one_video['segments'][:, 0])
            results['time_end'].append(result_one_video['segments'][:, 1])
            results['label'].append(result_one_video['labels'])
            results['score'].append(result_one_video['scores'])
        
    results['time_start'] = torch.cat(results['time_start']).numpy()
    results['time_end'] = torch.cat(results['time_end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    # create dataframe
    df = pd.DataFrame(results)
    df.sort_values(by=['video_id','time_start','time_end'], ascending=[True,True,True], inplace=True)
    df.to_csv(os.path.join(args.output_dir, "prediction.csv"), index=False)
