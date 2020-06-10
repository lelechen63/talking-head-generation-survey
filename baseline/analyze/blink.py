import sys
sys.path.insert(0, '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid')

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import random

from util.util import eye_openrate
from data.get_landmark import get_keypoints

import pdb

def get_landmark(paths, root, dataset):
    if dataset == 'grid':
        video_path = os.path.join(root, 'align', paths[0], paths[1]+'_crop.mp4')
        landmark_path = os.path.join(root, 'align', paths[0], paths[1]+'_original.npy')

    return landmark_path, video_path

def image_to_video(sample_dir, video_name):
    # command = 'ffmpeg -framerate 25  -i ' + sample_dir +  'landmark_%05d.jpg -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  'landmark_%05d.jpg -c:v libx264 -y -vf format=yuv420p ' + video_name 
    #ffmpeg -framerate 25 -i real_%d.png -c:v libx264 -y -vf format=yuv420p real.mp4
    print (command)
    os.system(command)

def plot_open_rate(landmarks, fig_path):
    openrates = eye_openrate(landmarks)
    plt.figure()
    plt.plot(list(range(len(openrates))), openrates)
    plt.savefig(fig_path)

def plot_landmarks(landmarks, fig_dir, lmark_path, video_name):
    for lmark_id, lmark in enumerate(landmarks):
        lmark_img = get_keypoints(lmark, (256, 256))
        lmark_img.save(lmark_path % lmark_id)
    image_to_video(fig_dir, video_name)

def get_figures():
    # files
    landmark_path = 'align/s1/lgif2p_original.npy'
    landmarks = np.load(landmark_path)
    fig_dir = 'analyze/blink_store/grid/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    lmark_path = os.path.join(fig_dir, 'landmark_%05d.jpg')
    fig_path = os.path.join(fig_dir, 'blink.png')
    video_name = os.path.join(fig_dir, 'landmark.mp4')

    # plot
    plot_open_rate(landmarks, fig_path)
    plot_landmarks(landmarks, fig_dir, lmark_path, video_name)

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    real_video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            real_video.append(frame)
        else:
            break

    return real_video

# given landmarks, find blinking motion by calculating open rate of eyes
def find_blink(lmarks, thre=18):
    openrates = eye_openrate(lmarks)
    # select non blink frame list 
    close_dict, open_dict = {}, {}
    non_blink_count = 0
    cur_ele = None
    non_blink_list = []
    non_blink_sel = openrates > thre
    for frame, ele in enumerate(non_blink_sel):
        if cur_ele is None:
            cur_ele = ele
            non_blink_count += 1
            non_blink_list.append(frame)
        elif ele == cur_ele:
            non_blink_count += 1
            non_blink_list.append(frame)
        else:
            # store
            if cur_ele == 0:
                cur_dict = close_dict
            else:
                cur_dict = open_dict
            if non_blink_count not in cur_dict:
                cur_dict[non_blink_count] = []
            cur_dict[non_blink_count].append(non_blink_list)
            # reset
            cur_ele = ele
            non_blink_count = 1
            non_blink_list = [frame]
    # final store
    if cur_ele == 0:
        cur_dict = close_dict
    else:
        cur_dict = open_dict
    if non_blink_count not in cur_dict:
        cur_dict[non_blink_count] = []
    cur_dict[non_blink_count].append(non_blink_list)
    # select blink frames
    sel_frames = []
    for close_count in close_dict:
        for frame_list in close_dict[close_count]:
            if len(frame_list) < 2:
                continue
            sel_ids = [frame_list[0]]
            if frame_list[-1] > max(sel_ids[0]+1, 6):
                sel_ids.append(frame_list[-1])
            # get frames
            for sel_id in sel_ids:
                if sel_id < 2:
                    sel_frames.append(np.arange(0,6))
                elif sel_id >= lmarks.shape[0] - 3:
                    sel_frames.append(np.arange(lmarks.shape[0]-6, lmarks.shape[0]))
                else:
                    sel_frames.append(np.arange(sel_id-2, sel_id+4))
    # select non blink frames (close)
    sel_frames_nonblink = []
    non_blink_count = list(close_dict.keys())
    non_blink_count = sorted(non_blink_count)[::-1]
    for count in non_blink_count:
        if count < 6:
            break
        for frame_list in close_dict[count]:
            start_point = frame_list[0]
            for start_point in range(frame_list[0], frame_list[-1], 6):
                if frame_list[-1] - start_point + 1 < 5:
                    continue
                if start_point < lmarks.shape[0] - 5:
                    sel_frames_nonblink.append(np.arange(start_point, start_point+6))
                else:
                    sel_frames_nonblink.append(np.arange(lmarks.shape[0]-6, lmarks.shape[0]))
    # select non blink frames (open)
    non_blink_count = list(open_dict.keys())
    non_blink_count = sorted(non_blink_count)[::-1]
    for count in non_blink_count:
        if count < 6:
            break
        if len(sel_frames_nonblink) >= len(sel_frames):
            break
        for frame_list in open_dict[count]:
            start_point = frame_list[0]
            for start_point in range(frame_list[0], frame_list[-1], 6):
                if start_point < lmarks.shape[0] - 5:
                    sel_frames_nonblink.append(np.arange(start_point, start_point+6))
                else:
                    sel_frames_nonblink.append(np.arange(lmarks.shape[0]-6, lmarks.shape[0]))

    return random.sample(sel_frames_nonblink, k=min(len(sel_frames_nonblink), len(sel_frames))), sel_frames

# find blink and save
def check_openrate():
    # files
    dataset = 'grid'
    root = 'grid'
    pickle_file = 'grid/pickle/test_audio2lmark_grid.pkl'
    with open(pickle_file, 'rb') as f:
        p_d = pickle.load(f)
    p_d = p_d[:20]
    # check blink
    fig_dir = 'analyze/blink_store/blink_check/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    for paths in tqdm(p_d):
        fig_path = os.path.join(fig_dir, '{}_{}'.format(paths[0], paths[1]))
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        # openrate
        lmark_path, video_path = get_landmark(paths, root, dataset)
        video = read_video(video_path)
        lmarks = np.load(lmark_path)
        # find blink
        non_blink_frames, blink_frames = find_blink(lmarks)
        for list_id, frame_list in enumerate(blink_frames):
            fig_file = os.path.join(fig_path, str(list_id))
            if not os.path.exists(fig_file):
                os.makedirs(fig_file)
            for f in frame_list:
                fig_file_f = os.path.join(fig_file, str(f)+'.png')
                cv2.imwrite(fig_file_f, video[f])

# separate blink frame and non-blink frame
def blink_frame_csv():
    # files
    dataset = 'grid'
    root = 'grid'
    pickle_file = 'grid/pickle/train_audio2lmark_grid.pkl'
    with open(pickle_file, 'rb') as f:
        p_d = pickle.load(f)
    # check blink
    csv_dir = 'analyze/blink_store/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_file = os.path.join(csv_dir, 'blink_train.pkl')
    csv_dict = {'paths':[], 'frames':[], 'labels':[]}
    for paths in tqdm(p_d):
        # openrate
        lmark_path, video_path = get_landmark(paths, root, dataset)
        lmarks = np.load(lmark_path)
        # find blink
        non_blink_frames, blink_frames = find_blink(lmarks)
        csv_dict['paths'] += [video_path]*(len(non_blink_frames)+len(blink_frames))
        csv_dict['frames'] += non_blink_frames
        csv_dict['frames'] += blink_frames
        csv_dict['labels'] += [0]*len(non_blink_frames)
        csv_dict['labels'] += [1]*len(blink_frames)
        pdb.set_trace()
    # store
    with open(csv_file, 'wb') as f:
        pickle.dump(csv_dict, f)

if __name__ == '__main__':
    # get_figures()
    check_openrate()
    # save_openrate_csv()