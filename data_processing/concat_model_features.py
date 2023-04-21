import numpy as np
import os, glob
from argparse import ArgumentParser

def create_args():
    parser = ArgumentParser(description="")
    parser.add_argument("--feature_dir_mvit", type=str, default="/workspace/AICity2023/data/feature_A2_MVit_concat_3view_mean5fold_1s")
    parser.add_argument("--feature_dir_x3d", type=str, default="/workspace/AICity2023/data/featureA2_14_03_concat_3view")
    parser.add_argument("--output_dir", type=str, default="/workspace/AICity2023/data/extract_feature_A2_MVit_mean5fold_1s")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cfg = create_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    for user_dir in sorted(glob.glob(f"{cfg.feature_dir_mvit}/*")):
        user_name = user_dir.split("/")[-1]
        out_path = os.path.join(cfg.output_dir, user_name)
        os.makedirs(out_path, exist_ok=True)

        feature_dict = {}
        dim_min = {}
        for _idx, video_path_1 in enumerate(sorted(glob.glob(f"{user_dir}/*.npy"))):
            # check the order of view in folder
            video_path_2 = os.path.join(cfg.feature_dir_x3d, user_name)
            video_name = video_path_1.split("/")[-1]
            video_path_2 = os.path.join(video_path_2, video_name)
            print(video_path_2)
        #     video_name = "user_id" + view_name.split("user_id")[-1]
            feature_1_npy = np.load(video_path_1)
            feature_2_npy = np.load(video_path_2)
            min_len = min(feature_1_npy.shape[0], feature_2_npy.shape[0])
            print(feature_1_npy[:min_len].shape, feature_2_npy[:min_len].shape)
            feature_concat_npy = np.concatenate((feature_1_npy[:min_len], feature_2_npy[:min_len]), 1)
            print(feature_concat_npy.shape)
            print(f"Video name: {video_path_2} | Shape: {feature_concat_npy.shape}")
            # print(f"Min dim:", dim_min[video_name])
            # print(os.path.join(out_path, video_name))
            np.save(os.path.join(out_path, video_name), feature_concat_npy)

        # for video_name, feature_list in feature_dict.items():
        #     # handle the duration difference between views
        #     feature_list_truncated = []
        #     for _f in feature_list:
        #         feature_list_truncated.append(_f[:dim_min[video_name], :])

        #     feature_3view = np.concatenate(feature_list_truncated, axis=1)
        #     print(f"Video name: {video_name} | Shape: {feature_3view.shape}")
        #     # print(f"Min dim:", dim_min[video_name])
        #     # print(os.path.join(out_path, video_name))
        #     np.save(os.path.join(out_path, video_name), feature_3view)
        
        # # raise Exception
             
