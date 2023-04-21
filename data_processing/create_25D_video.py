import cv2
import os
import numpy as np
import time
import datetime
import argparse
import imageio

from pathlib import Path
from tqdm import tqdm, auto
from multiprocessing import pool



print(os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument("--step", default=3, type=int)
parser.add_argument("--worker", default=4, type=int)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--src", default=None, type=int)
parser.add_argument("--dst", default=None, type=int)
args = parser.parse_args()

step = args.step
ori_videos_dir = args.src
gen_videos_dir = args.dst

# ori_videos_dir = "AICity2023-SetA2-zipfile"
# gen_videos_dir = "A2_raw_2.5D"

list_file_paths = list()

for class_id in os.listdir(ori_videos_dir):
    class_id_dir = os.path.join(ori_videos_dir, class_id)
    if not os.path.isdir(class_id_dir):
        continue
    for file_name in os.listdir(class_id_dir):
        if Path(file_name).suffix == ".MP4":
            file_path = os.path.join(class_id_dir, file_name)
            list_file_paths.append(file_path)

def gen(video_idx):
    T1 = time.time()

    file_path = list_file_paths[video_idx]
    file_rel_path = os.path.relpath(file_path, ori_videos_dir)
    # tqdm.write(f"{file_rel_path}")
    gen_file_path = os.path.join(gen_videos_dir, file_rel_path)
    os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)

    video = None
    # cap = cv2.VideoCapture(file_path)
    # cap.set(cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE)
    # fps = 30
    # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # del cap

    cap = imageio.get_reader(file_path)
    num_frames = cap.count_frames()
    fps = cap.get_meta_data()["fps"]

    index = 0
    max_index = 0
    count_frames = 0
    dict_frames = {}
    for index in range(num_frames):
        get_index = min(num_frames - 1, index)
        if get_index not in dict_frames.keys():
            try:
                dict_frames[get_index] = cv2.cvtColor(cap.get_data(get_index), cv2.COLOR_RGB2GRAY)
                max_index = get_index
            except:
                dict_frames[get_index] = cv2.cvtColor(cap.get_data(max_index), cv2.COLOR_RGB2GRAY)
                # dict_frames[get_index] = dict_frames[max_index]
        frame1 = dict_frames[get_index]

        get_index = min(num_frames - 1, index + step//3)
        if get_index not in dict_frames.keys():
            try:
                dict_frames[get_index] = cv2.cvtColor(cap.get_data(get_index), cv2.COLOR_RGB2GRAY)
                max_index = get_index
            except:
                dict_frames[get_index] = cv2.cvtColor(cap.get_data(max_index), cv2.COLOR_RGB2GRAY)
                # dict_frames[get_index] = dict_frames[max_index]
        frame2 = dict_frames[get_index]

        get_index = min(num_frames - 1, index + 2*step//3)
        if get_index not in dict_frames.keys():
            try:
                dict_frames[get_index] = cv2.cvtColor(cap.get_data(get_index), cv2.COLOR_RGB2GRAY)
                max_index = get_index
            except:
                dict_frames[get_index] = cv2.cvtColor(cap.get_data(max_index), cv2.COLOR_RGB2GRAY)
                # dict_frames[get_index] = dict_frames[max_index]
        frame3 = dict_frames[get_index]

        frame = np.stack([frame1, frame2, frame3], axis=2)
        if video is None:
            video_height, video_width = frame.shape[:2]
            video = cv2.VideoWriter(gen_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (video_width, video_height))
        video.write(frame)

        # tqdm.write(f"{index:5}-{num_frames}-{dict_frames.keys()}")
        if index in dict_frames.keys() and index + step < num_frames:
            del dict_frames[index]
        count_frames += 1

        # if index >= num_frames:
        #     break
        # print(f"{file_rel_path:60} - {index}/{num_frames}")

    # cv2.destroyAllWindows()
    video.release()

    T2 = time.time()
    tqdm.write(f"{video_idx}\t - {file_rel_path:60} - {datetime.timedelta(seconds=round(T2 - T1))} - {count_frames}/{index}/{num_frames}")

num_files = len(list_file_paths)
# for video_idx in tqdm(range(num_files)):
#     gen(video_idx)

start_idx = args.start
end_idx = args.end if args.end else num_files
results = pool.ThreadPool(args.worker).imap(gen, list(range(start_idx, end_idx)))
pbar = auto.tqdm(results, total=end_idx - start_idx)
for _ in pbar:
    pass