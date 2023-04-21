import pandas as pd
import os
from argparse import ArgumentParser

def create_args():
    parser = ArgumentParser(description="")
    parser.add_argument("--dataset_path", type=str, default="/workspace/AICity2023/data/SetA2")
    parser.add_argument("--prediction_path", type=str, default="/workspace/AICity2023/TriDet-master/results/MVit/mvit_concat_3view_mean5fold_v3")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cfg = create_args()

    df = pd.read_csv(f"{cfg.prediction_path}/prediction.csv")
    format_df = pd.read_csv(f"{cfg.dataset_path}/video_ids.csv")

    # create mapping between name and id video
    name2id = dict()
    format_df = format_df[['video_id','video_files']]
    format_df['video_files'] = format_df['video_files'].apply(lambda x: "user_id" + x.split("user_id")[-1])
    # format_df = dict(format_df)
    # print(format_df)

    # post-processing: convert action former prediction -> submission format
    df['time_start'] = df['time_start'].apply(lambda x: round(x))
    df['time_end'] = df['time_end'].apply(lambda x: round(x))
    df['label'] = df['label'] + 1
    df = df.drop(columns=['score'])

    # merge id
    df = df.merge(format_df, how='left', left_on='video_id', right_on='video_files')
    df = df[['video_id_y','label','time_start','time_end']]
    df.to_csv(os.path.join(cfg.prediction_path, "submission.txt"), index=False, header=False, sep=" ")