import numpy as np
import pandas as pd

import os, shutil, glob
from argparse import ArgumentParser

def create_args():
    parser = ArgumentParser(description="")
    parser.add_argument("--label_df_path", type=str, default="../data/SetA1/processed_label_csv")
    parser.add_argument("--output_path", type=str, default="../data/train_val_split_3views_more_cls0_2.5D")

    args = parser.parse_args()
    return args

NUM_FOLD = 5
MAP_UNIT_TIME = [3600,60,1]
VIEW = ['Dashboard', 'Rearview', 'Right_side_window']

if __name__ == "__main__":
    cfg = create_args()
    for fold in range(NUM_FOLD):
        os.makedirs(f"{cfg.output_path}/fold{fold}", exist_ok=True)
        
        for view in VIEW:
            os.makedirs(f"{cfg.output_path}/fold{fold}/{view}", exist_ok=True)
            f_train = open(f"{cfg.output_path}/fold{fold}/{view}/train.csv", "a")
            f_val = open(f"{cfg.output_path}/fold{fold}/{view}/val.csv","a")
            f_test = open(f"{cfg.output_path}/fold{fold}/{view}/test.csv","a")

            files = sorted(os.listdir(cfg.label_df_path))
            for idx, user in enumerate(files):
                df = pd.read_csv(f"{cfg.label_df_path}/{files[idx]}")
                for i in range(len(df)):
                    filename = df['Filename'][i]
                    filename = filename.replace("Rear_view", "Rearview")
                    if view in filename:
                        # original label
                        path_to_clip = f"/workspace/AICity2023/data/A1_train_2.5D/{df['Class'][i]}/{df['Filename'][i]}_time_{df['Start Time'][i].strip()}_{df['End Time'][i].strip()}.MP4"
                        cls = df['Class'][i]
                        if idx % NUM_FOLD == fold:
                            f_val.write(f"{path_to_clip} {cls}\n")
                            f_test.write(f"{path_to_clip} {cls}\n")
                        else:
                            f_train.write(f"{path_to_clip} {cls}\n")
                            
                        # extracted label class 0
                        if i < len(df) - 1:
                            time_start = sum([a*b for a,b in zip(MAP_UNIT_TIME, map(int,df['End Time'][i].split(':')))]) + 1
                            time_end = sum([a*b for a,b in zip(MAP_UNIT_TIME, map(int,df['Start Time'][i+1].split(':')))]) - 1
                            if time_start < time_end:
                                path_to_clip = f"/workspace/AICity2023/data/SetA1_clip/0/{df['Filename'][i]}_time_{df['End Time'][i].strip()}_{df['Start Time'][i+1].strip()}.MP4"
                                cls = 0
                                if idx % NUM_FOLD == fold:
                                    f_val.write(f"{path_to_clip} {cls}\n")
                                    f_test.write(f"{path_to_clip} {cls}\n")
                                else:
                                    f_train.write(f"{path_to_clip} {cls}\n")
    #                 # check filename outliers
    #                 flag = False
    #                 for _v in VIEW:
    #                     if _v in filename: flag = True
    #                 if flag is False: print(df['Filename'][i])
                            
            f_train.close()
            f_val.close()
            f_test.close()
