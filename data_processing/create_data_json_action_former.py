import pandas as pd
import glob
import os
import json
from argparse import ArgumentParser

MAP_UNIT_TIME = [3600,60,1]

def create_args():
    parser = ArgumentParser(description="")
    parser.add_argument("--use_concat_view", default=True, type=bool)
    parser.add_argument("--split_csv_dir", default="/workspace/AICity2023/data/train_val_split_3views_without_cls0", type=str)
    parser.add_argument("--output_dir", default="/workspace/AICity2023/actionformer/data/aic_2023_SetA1_2model_concat_3view", type=str)
    parser.add_argument("--name_folder_raw", default="SetA1", type=str, help="where MP4 videos store")
    parser.add_argument("--name_folder_extract", default="feature_A1_2model_concat_3view", type=str, help="where video features store")
    
    args = parser.parse_args()
    return args

def create_json_format_per_view(file_train, result, type_split, config):
    filename_mp4_prev = ""
    path_prev = ""
    list_annotations = []

    for _idx, line in enumerate(file_train):
        line = line.strip()
        label = line.split(",")[1]
        path = line.split(",")[0].replace("_clip", "")
        
        name_video = path.split("/")[-1]
        path = path.split("/")[:-2]
        path = '/'.join(path)
        
        label_time = name_video.split("_time_")[1].replace(".MP4", "")
        name_video = name_video.split("_time_")[0]

        filename_mp4 = name_video.replace("User", "user")
        if "Dash_board_user" in filename_mp4:
            filename_mp4 = filename_mp4.replace("Dash_board_user", "Dashboard_user")
        if "Rearview_user" in filename_mp4:
            filename_mp4 = filename_mp4.replace("Rearview_user", "Rear_view_user")
        if "Rightside_window_user" in filename_mp4:
            filename_mp4 = filename_mp4.replace("Rightside_window_user", "Right_side_window_user")

        user_id = filename_mp4.split("_")[-2]
        prefix_id = filename_mp4.split("_")[-1]

        split_filename_mp4 = filename_mp4.split("_")
        split_filename_mp4[-1] = "NoAudio"
        split_filename_mp4.append(prefix_id)
        filename_mp4 = "_".join(split_filename_mp4)
        filename_mp4 += ".MP4"

        if config.use_concat_view:
            filename_mp4 = "user_id" + filename_mp4.split("user_id")[-1]
        path = os.path.join(path, "user_id_"+user_id, filename_mp4)
        path = path.replace(config.name_folder_raw, config.name_folder_extract)
        path = path + ".npy" # where video features store

        start_time = label_time.split("_")[0]
        end_time = label_time.split("_")[1]
        start_time = sum([a*b for a,b in zip(MAP_UNIT_TIME, map(int,start_time.split(':')))])
        end_time = sum([a*b for a,b in zip(MAP_UNIT_TIME, map(int,end_time.split(':')))])

        # print("path", path)
        # print("label_time", label_time)
        # print("start_time", start_time)
        # print("end_time", end_time)
        # print("filename_mp4", filename_mp4)
        # print(label)

        if filename_mp4_prev == "":
            annotations = {}
            annotations["segment"] = [start_time, end_time]
            annotations["label"] = label
            annotations["label_id"] = int(label) -1
            list_annotations.append(annotations)
            filename_mp4_prev = filename_mp4
        else:
            if line != "":
                annotations = {}
                annotations["segment"] = [start_time, end_time]
                annotations["label"] = label
                annotations["label_id"] = int(label) -1

                if filename_mp4_prev!=filename_mp4:
                    result["database"][filename_mp4_prev] = {}
                    result["database"][filename_mp4_prev]["duration"] = list_annotations[-1]["segment"][1] # ???? Mr.Tung's code: logical mistake
                    result["database"][filename_mp4_prev]["subset"] = type_split
                    result["database"][filename_mp4_prev]["url"] = path_prev
                    result["database"][filename_mp4_prev]["resolution"] = "1920x1080"
                    result["database"][filename_mp4_prev]["annotations"] = list_annotations
                    list_annotations = [annotations]
                else:
                    list_annotations.append(annotations)
            else:
                prin("WRONGGGGG")
            filename_mp4_prev = filename_mp4        
        path_prev = path

    if len(list_annotations) > 0:
        result["database"][filename_mp4_prev] = {}
        result["database"][filename_mp4_prev]["duration"] = list_annotations[-1]["segment"][1]
        result["database"][filename_mp4_prev]["subset"] = type_split
        result["database"][filename_mp4_prev]["url"] = path_prev
        result["database"][filename_mp4_prev]["resolution"] = "1920x1080"
        result["database"][filename_mp4_prev]["annotations"] = list_annotations

if __name__ == "__main__":
    config = create_args()

    list_name_fold = os.listdir(config.split_csv_dir)
    for name_fold in list_name_fold:
        path_fold = os.path.join(config.split_csv_dir, name_fold)
        list_view_folder = os.listdir(path_fold)
        for view_folder in list_view_folder:
            if config.use_concat_view:
                if "Dashboard" not in view_folder:
                    continue
            path_view_folder = os.path.join(path_fold, view_folder)

            # print(config.output_dir)
            if config.use_concat_view:
                path_file_json = os.path.join(config.output_dir, f"{name_fold}")
            else:
                path_file_json = os.path.join(config.output_dir, f"{name_fold}/{view_folder}")
            # print(path_file_json)
            os.makedirs(path_file_json, exist_ok=True)
            path_file_json = os.path.join(path_file_json, "AICity2023.json")

            path_train_csv = os.path.join(path_view_folder, "train.csv")
            path_test_csv = os.path.join(path_view_folder, "test.csv")
            file_train = open(path_train_csv, 'r')
            file_test = open(path_test_csv, 'r')

            result = {"database": {}}
            create_json_format_per_view(file_train, result, "training", config)
            create_json_format_per_view(file_test, result, "validation", config)

            print("OUTPUT: ", path_file_json)
            with open(path_file_json, "w") as write_file:
                json.dump(result, write_file)
            # break
        # break
