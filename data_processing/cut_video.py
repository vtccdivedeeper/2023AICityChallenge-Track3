# Import everything needed to edit video clips
from moviepy.editor import *
import pandas as pd
import os
from argparse import ArgumentParser

def create_args():
    parser = ArgumentParser(description="")
    parser.add_argument("--input_dir", type=str, default="../data/SetA1")
    parser.add_argument("--output_dir", type=str, default="../data/SetA1_clip")
    
NUM_CLASS = 16
MAP_UNIT_TIME = [3600,60,1]
# start_idx = 13
# end_idx = 14

def cut_video(clip1, time_start, time_end, path_video):
    # getting width and height of clip 1
    # w1 = clip1.w
    # h1 = clip1.h
    # print("Width x Height of clip 1 : ", end = " ")
    # print(str(w1) + " x ", str(h1))
    # print("---------------------------------------")
    
    # resizing video downsize 50 %
    clip2 = clip1.subclip(time_start, time_end).resize((512, 512))

    # getting width and height of clip 1
    # w2 = clip2.w
    # h2 = clip2.h
    # print("Width x Height of clip 2 : ", end = " ")
    # print(str(w2) + " x ", str(h2))
    # print("---------------------------------------")
    clip2.write_videofile(path_video)

if __name__ == "__main__":
    cfg = create_args()

    #create folder data
    if not os.path.isdir(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    else: 
        print("folder already exists.")

    for i in range(NUM_CLASS):
        data_dir = f'{cfg.output_dir}/{str(i)}'
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
            print("Create folder: ", data_dir)
        else:
            print(data_dir, "folder already exists.")

    for user_idx, user_name in enumerate(sorted(os.listdir(cfg.input_dir))):
        # if (user_idx < start_idx) or (user_idx >= end_idx):
        #     continue
        if "user" not in user_name:
            continue

        # track_file = open(f"../data/track_clipvideo_{start_idx}_{end_idx}.txt", "a")
        print("="*20 + f"{user_name}" + "="*20)
        # track_file.write("="*20 + f"{user_name}" + "="*20 + "\n")
        # track_file.close()

        path_folder = '{}/{}'.format(cfg.input_dir,user_name)
        path_csv = '{}/processed_label_csv/{}.csv'.format(cfg.input_dir, user_name)
        df = pd.read_csv(path_csv)

        filename = "Unknown"
        cnt = 0
        for i in range(len(df)):
            if df['Filename'][i] != filename:
                # print(f"Summarize view: {filename} | Total clip: {cnt}")
                # track_file = open(f"../data/track_clipvideo_{start_idx}_{end_idx}.txt", "a")
                # track_file.write(f"Summarize view: {filename} | Total clip: {cnt} \n")
                # track_file.close()

                # move on next video
                filename = df['Filename'][i] # example: Dashboard_User_id_71436_5
                cnt = 0

                # modify filename for the inconsistence
                filename_mp4 = filename.replace("User", "user")
                if "Dash_board_user" in filename_mp4:
                    filename_mp4 = filename_mp4.replace("Dash_board_user", "Dashboard_user")
                if "Rearview_user" in filename_mp4:
                    filename_mp4 = filename_mp4.replace("Rearview_user", "Rear_view_user")
                if "Rightside_window_user" in filename_mp4:
                    filename_mp4 = filename_mp4.replace("Rightside_window_user", "Right_side_window_user")

                full_clip = VideoFileClip("{}/{}_NoAudio_{}.MP4".format(path_folder, "_".join(filename_mp4.split("_")[:-1]), filename_mp4.split("_")[-1]))

            time_start = sum([a*b for a,b in zip(MAP_UNIT_TIME, map(int,df['Start Time'][i].split(':')))])
            time_end = sum([a*b for a,b in zip(MAP_UNIT_TIME, map(int,df['End Time'][i].split(':')))])
            path_video = '{}/{}/{}_time_{}_{}.MP4'.format(cfg.output_dir, df['Class'][i], filename, df['Start Time'][i].strip(), df['End Time'][i].strip())
            cut_video(full_clip, time_start, time_end, path_video)

            # extract clip class 0
            if i < len(df) - 1:
                time_start = sum([a*b for a,b in zip(MAP_UNIT_TIME, map(int,df['End Time'][i].split(':')))]) + 1
                time_end = sum([a*b for a,b in zip(MAP_UNIT_TIME, map(int,df['Start Time'][i+1].split(':')))]) - 1
                if time_start < time_end:        
                    # print('time_start', time_start, 'time_end', time_end)
                    path_video = '{}/{}/{}_time_{}_{}.MP4'.format(cfg.output_dir, 0, filename, df['End Time'][i].strip(), df['Start Time'][i+1].strip())
                    cut_video(full_clip, time_start, time_end, path_video)
            cnt += 1

