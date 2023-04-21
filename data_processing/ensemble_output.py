import pandas as pd
import os
import numpy as np
import torch
from nms import batched_nms

def create_nsm_format(pred_df1, pred_df2):
    pred_dict = []
    video_names = pred_df1["video_id"].unique()
    for video_name in video_names:
        video_dict = dict()

        df_video1 = pred_df1[pred_df1.video_id == video_name]
        df_video2 = pred_df2[pred_df2.video_id == video_name]

        # df1
        start = list(df_video1['time_start'])
        end = list(df_video1['time_end'])
        video_dict['video_id'] = video_name
        video_dict['segments'] = list(zip(start,end))
        video_dict['labels'] = list(df_video1['label'])
        video_dict['scores'] = list(df_video1['score'])
        
        # df2
        start = list(df_video2['time_start'])
        end = list(df_video2['time_end'])
        video_dict['video_id'] = video_name
        video_dict['segments'].extend(list(zip(start,end)))
        video_dict['labels'].extend(list(df_video2['label']))
        video_dict['scores'].extend(list(df_video2['score']))

        video_dict['segments'] = torch.Tensor(video_dict['segments'])
        video_dict['labels'] = torch.Tensor(video_dict['labels'])
        video_dict['scores'] = torch.Tensor(video_dict['scores'])

        pred_dict.append(video_dict)

    return pred_dict

def nms_process(nms_format):
    processed_results = []
    for results_per_vid in nms_format:
        vidx = results_per_vid['video_id']
        segs = results_per_vid['segments']
        scores = results_per_vid['scores']
        labels = results_per_vid['labels']

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
        processed_results.append(
            {'video_id' : vidx,
                'segments' : segs,
                'scores'   : scores,
                'labels'   : labels}
        )
    return processed_results

test_cfg = {
    'duration_thresh': 0.05,
    'ext_score_file': None,
    'iou_threshold': 0.3,
    'max_seg_num': 500,
    'min_score': 0.8,
    'multiclass_nms': True,
    'nms_method': 'soft',
    'nms_sigma': 0.5,
    'pre_nms_thresh': 0.001,
    'pre_nms_topk': 2000,
    'voting_thresh': 0.7
}

if __name__ == "__main__":
    RAW_DATA_SETA2 = "/workspace/AICity2023/data/SetA2"
    OUTPUT = "/workspace/AICity2023/actionformer/results/ensemble_v2"
    RAW_PREDICTION1 = "/workspace/AICity2023/actionformer/results/X3D/20_03_concat_3view_v6"
    RAW_PREDICTION2 = "/workspace/AICity2023/actionformer/results/MVit/mvit_concat_3view_v6"

    df1 = pd.read_csv(f"{RAW_PREDICTION1}/prediction.csv")
    df2 = pd.read_csv(f"{RAW_PREDICTION2}/prediction.csv")
    format_df = pd.read_csv(f"{RAW_DATA_SETA2}/video_ids.csv")

    nms_format = create_nsm_format(df1, df2)
    results_final = nms_process(nms_format)

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
    # post-processing: convert action former prediction -> submission format
    df['time_start'] = df['time_start'].apply(lambda x: round(x))
    df['time_end'] = df['time_end'].apply(lambda x: round(x))
    df['label'] = df['label'] + 1
    df['label'] = df['label'].astype(np.int32)
    df = df.drop(columns=['score'])

    # create mapping between name and id video
    name2id = dict()
    format_df = format_df[['video_id','video_files']]
    format_df['video_files'] = format_df['video_files'].apply(lambda x: "user_id" + x.split("user_id")[-1])
    # format_df = dict(format_df)
    # print(format_df)

    # merge id
    df = df.merge(format_df, how='left', left_on='video_id', right_on='video_files')
    df = df[['video_id_y','label','time_start','time_end']]
    df.sort_values(by=['video_id_y','time_start','time_end'], ascending=[True,True,True], inplace=True)

    # save
    os.makedirs(OUTPUT, exist_ok=True)
    with open(f"{OUTPUT}/config.txt", "w") as f:
        f.write(RAW_PREDICTION1)
        f.write("\n")
        f.write(RAW_PREDICTION2)

    df.to_csv(os.path.join(OUTPUT, "submission.txt"), index=False, header=False, sep=" ")