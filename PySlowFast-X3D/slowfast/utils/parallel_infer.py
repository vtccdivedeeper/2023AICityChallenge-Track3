import cv2
import os
import time
import torch
from multiprocessing import Process, Queue, Manager, Lock
import albumentations as A
from scipy.special import softmax
import numpy as np
import pandas as pd
from moviepy.editor import *

image_size_list = [448, 288, 384, 224, 608]
# image_size_list = [448, 288, 384, 224, 224]
test_transform_map = {}
use_TTA = True

def preprocess(frame, image_size, TTA):
    frame = test_transform_map[image_size](image=frame)["image"]
    v_frame = None
    if TTA:
        v_frame = A.VerticalFlip(p=1)(image=frame)["image"]
    return frame, v_frame


def extract_frames(video_path, cfg, use_lib="cv2"):
    sampling_rate = cfg.DATA.SAMPLING_RATE
    if use_lib == "cv2":
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        num_extracted_frame = 0
        all_inputs = [] # ~ 1 video
        _input = [] # ~ 1 clip
        cnt_frames = 0
        # for i in range(0, total_frames, sampling_rate):
        #     if i > 5000: break
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        #     ret, frame = cap.read()
        #     if ret:
        #         # frames.append(frame)
        #         continue
        #     else:
        #         break
        stride = cfg.DATA.STRIDE
        list_queue_frame = []
        for i in range(total_frames):
            cap.grab()
            if i % sampling_rate == 0:
                # print(i)
                ret, frame = cap.retrieve()
                if ret:
                    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed = cv2.resize(frame_processed, (cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE), interpolation = cv2.INTER_AREA)
                    # _input.append(frame_processed)
                    # cnt_frames += 1
                    num_extracted_frame += 1
                    if (i % (stride *sampling_rate) ==0 and len(list_queue_frame) == cfg.DATA.NUM_FRAMES + stride):
                        # print('index', i)
                        for i_stride in range(stride):
                            list_queue_frame.pop(0)
                        _input = list_queue_frame
                        _input = torch.tensor(np.array(_input)).float()
                        _input = (_input / 255.0 - torch.tensor(cfg.DATA.MEAN)) / torch.tensor(cfg.DATA.STD)
                        _input = _input.permute(3, 0, 1, 2)
                        # list_queue_frame.append(_input)
                        all_inputs.append(_input)
                        #reset
                        # cnt_frames = 0

                    list_queue_frame.append(frame_processed)
                    # if cnt_frames == cfg.DATA.NUM_FRAMES:
                    #     _input = torch.tensor(np.array(_input)).float()
                    #     _input = (_input / 255.0 - torch.tensor(cfg.DATA.MEAN)) / torch.tensor(cfg.DATA.STD)
                    #     _input = _input.permute(3, 0, 1, 2)
                    #     # list_queue_frame.append(_input)
                    #     all_inputs.append(_input)

                    #     # reset
                    #     _input = []
                    #     cnt_frames = 0
                else: break

        cap.release()

    elif use_lib == "moviepy":
        cap = VideoFileClip(video_path).resize(height=cfg.DATA.TEST_CROP_SIZE, width= cfg.DATA.TEST_CROP_SIZE)
        total_frames = int(cap.fps * cap.duration)
        frames = []
        all_inputs = []
        _input = []
        cnt_frames = 0
        num_extracted_frame = 0
        for i in range(0, total_frames, sampling_rate):
            # if i > 234: break
            frame_processed = cap.get_frame(i)
            # frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            # frame_processed = cv2.resize(frame_processed, (cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE), interpolation = cv2.INTER_AREA)
            _input.append(frame_processed)
            cnt_frames += 1
            num_extracted_frame += 1

            if cnt_frames == cfg.DATA.NUM_FRAMES:
                _input = torch.tensor(np.array(_input)).float()
                _input = (_input / 255.0 - torch.tensor(cfg.DATA.MEAN)) / torch.tensor(cfg.DATA.STD)
                _input = _input.permute(3, 0, 1, 2)
                all_inputs.append(_input)
                _input = []
                cnt_frames = 0


    print(f"Number of extracted frame: {num_extracted_frame}/{total_frames} | Number of clips: {len(all_inputs)}",)
    return all_inputs

for image_size in image_size_list:
    test_transform = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC, p=1)
    ])
    test_transform_map[image_size] = test_transform

def extract_worker(q):
    video_directory = "/data/azc/private_test/videos"
    video_ids = []
    batch_size = 12
    for root, dirs, files in os.walk(video_directory):
        for filename in files:
            video_path = os.path.join(root, filename)
            frames = extract_frames(video_path, 2*batch_size/3)
            # frames = extract_frames(video_path, batch_size)
            total_frames = len(frames)
            for i in range(len(q)):
                batch = []
                for j in range(total_frames):
                    # TTA with ratio 1/2
                    if j % 2 == 0:
                        frame, v_frame = preprocess(frames[j], image_size_list[i], use_TTA)
                        if use_TTA:
                            batch.append(v_frame)
                    else:
                        frame, v_frame = preprocess(frames[j], image_size_list[i], False)

                    batch.append(frame)
                len_batch = len(batch)
                if len_batch < batch_size:
                    for j in range(batch_size - len_batch):
                        frame, v_frame = preprocess(frames[len_batch - 1], image_size_list[i], use_TTA)
                        batch.append(frame)

                batch = np.array(batch).astype("float32")
                batch_queue = {"id": filename, "n": len_batch, "batch": batch}
                q[i].put(batch_queue, block=True, timeout=None)
    
    # end process
    for i in range(len(q)):
        q[i].put(None, block=True, timeout=None)

def predict_worker(q, d, id, exp, lock):
    models = load_model(exp, id)
    lock.release()
    while True:
        batch_queue = q[id].get()
        if batch_queue is None:
            break
        video_name = batch_queue["id"]
        total_frames = batch_queue["n"]
        batch = batch_queue["batch"]
        average_score = 0
        for i in range(len(models)):
#             logits = models[i].run(None, {'input': batch})[0]
#             probs = softmax(logits, axis=1)
#             scores = probs[:,1]
            scores = models[i].run(None, {'input': batch})[0]
            # print(scores.shape)
            for j in range(total_frames):
                average_score += scores[j]
        average_score /= (total_frames*len(models))
        d[video_name] = average_score

if __name__ == '__main__':
    manager = Manager()
    q = [Queue(maxsize=10), Queue(maxsize=10), Queue(maxsize=10), Queue(maxsize=10)]
    d = [manager.dict(), manager.dict(), manager.dict(), manager.dict()]
    extract_process = Process(target=extract_worker, args=(q,))
    lock_1 = Lock()
    lock_2 = Lock()
    lock_3 = Lock()
    lock_4 = Lock()

    lock_1.acquire(); lock_2.acquire(); lock_3.acquire(); lock_4.acquire()
    predict_1_process = Process(target=predict_worker, args=(q, d[0], 0, "exp_39", lock_1, ))
    predict_2_process = Process(target=predict_worker, args=(q, d[1], 1, "exp_40", lock_2,))
    predict_3_process = Process(target=predict_worker, args=(q, d[2], 2, "exp_43", lock_3,))
    predict_4_process = Process(target=predict_worker, args=(q, d[3], 3, "exp_49", lock_4,))
    # predict_5_process = Process(target=predict_worker, args=(q, d[4], 4, "B5",))
    # results.append({})

    # Load model
    predict_1_process.start()
    predict_2_process.start()
    predict_3_process.start()
    predict_4_process.start()
    # predict_5_process.start()
    # Start run
    lock_1.acquire(); lock_2.acquire(); lock_3.acquire(); lock_4.acquire()
    t1 = time.time()
    lock_1.release(); lock_2.release(); lock_3.release(); lock_4.release()
    extract_process.start()
    extract_process.join()
    predict_1_process.join()
    predict_2_process.join()
    predict_3_process.join()
    predict_4_process.join()
    # predict_5_process.join()
    t2 = time.time()
    print("Total processing time (Include extract frames and preprocessing): {}s".format(t2 - t1))

    model_ensemble = {}
    model_ensemble_fname = []
    model_ensemble_liveness_score = []
    for k, v in d[0].items():
        model_ensemble[k] = 0

    for i in range(len(d)):
        for k, v in d[i].items():
            model_ensemble[k] += v
    for k, v in model_ensemble.items():
        model_ensemble_fname.append(k)
        model_ensemble_liveness_score.append(v/len(d))

    model_ensemble_fname = np.array(model_ensemble_fname)
    model_ensemble_liveness_score = np.array(model_ensemble_liveness_score)
    model_ensemble_fname = np.expand_dims(model_ensemble_fname, 1)
    model_ensemble_liveness_score = np.expand_dims(model_ensemble_liveness_score, 1)
    # print(model_ensemble_fname.shape, model_ensemble_liveness_score.shape)
    df = pd.DataFrame(np.concatenate((model_ensemble_fname, model_ensemble_liveness_score), axis=1), columns=["fname", "liveness_score"])
    os.makedirs("/result", exist_ok=True)
    print(df)
    df.to_csv('/result/submission.csv', index=False)
