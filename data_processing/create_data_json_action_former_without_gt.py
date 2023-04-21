import pandas as pd
import numpy as np
import glob
import os
import json
from moviepy.editor import *
from argparse import ArgumentParser

def create_args():
    parser = ArgumentParser(description="")
    parser.add_argument("--use_concat_view", default=True, type=bool)
    parser.add_argument("--clip_duration", default=1, type=int)
    parser.add_argument("--dataset_dir", default="/workspace/AICity2023/data/SetA2", type=str)
    parser.add_argument("--feature_dir", default="/workspace/AICity2023/data/featureA2_2model_concat_3view", type=str)
    parser.add_argument("--output_dir", default="/workspace/AICity2023/actionformer/data/aic_2023_SetA2_2model_concat_3view", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    config = create_args()

    os.makedirs(config.output_dir, exist_ok=True)
    result = {"database": {}}
    split = "inference"

    for user_dir in sorted(glob.glob(f"{config.dataset_dir}/*")):
        if ".csv" in user_dir:
            continue

        for video_path in sorted(glob.glob(os.path.join(user_dir, "*.MP4"))):
            if config.use_concat_view:
                view_name = video_path.split("/")[-1]
                video_name = "user_id" + view_name.split("user_id")[-1]
                if video_name in result["database"].keys():
                    continue
            else:
                video_name = video_path.split("/")[-1]
            print(video_name)

            url = f"{config.feature_dir}/{user_dir.split('/')[-1]}/{video_name}.npy"
            feature = np.load(url)
            duration = int(feature.shape[0]*config.clip_duration)
            result["database"][video_name] = {"duration": duration,
                                             "subset": split,
                                             "url": url,
                                             "resolution": "1920x1080"}

    with open(os.path.join(config.output_dir, "AICity2023.json"), "w") as f:
        json.dump(result, f)