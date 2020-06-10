import sys
sys.path.insert(0, '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid')

import pickle
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import random

from util.util import rt_to_degree

import pdb

# get rotation for specific dataset
def get_rt(paths, root, dataset):
    if dataset == 'vox':
        audio_package = 'unzip/test_video'
        rt_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned_rt.npy")
        img_path = os.path.join(paths[0], paths[1], paths[2])
        video_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned.mp4")
    elif dataset == 'lrs':
        audio_package = 'test'
        rt_path = os.path.join(root, audio_package, paths[0], paths[1]+ '_rt.npy')
        img_path = os.path.join(paths[0], paths[1])
        video_path = os.path.join(root, audio_package, paths[0], paths[1]+"_crop.mp4")
    elif dataset == 'lrw':
        rt_path = os.path.join(paths[0]+"_rt.npy")
        img_path = paths[0]
        video_path = os.path.join(paths[0]+"_crop.mp4")
    elif dataset == 'crema':
        video_package = 'VideoFlash'
        rt_path = os.path.join(root, video_package, paths[0][:-10]+"_rt.npy")
        img_path = paths[0][:-10]
        video_path = os.path.join(root, video_package, paths[0][:-10]+"_crop.mp4")
    elif dataset == 'face':
        rt_path = os.path.join(root, paths[1])
        img_path = paths[0]
        video_path = os.path.join(root, paths[0])
    elif dataset == 'obama':
        rt_path = os.path.join(root, paths+'__rt2.npy')
        img_path = paths
        video_path = os.path.join(root, paths+'__crop2.mp4')
    return rt_path, img_path, video_path

# calculate degree from rotation and save
def save_degree_from_rt(pickle_file, save_file, root, dataset):
    # read in pickle
    with open(pickle_file, 'rb') as f:
        p_d = pickle.load(f)
        if dataset == 'crema':
            p_d = p_d[int(len(p_d)*0.8):]
    
    print('total len {}'.format(len(p_d)))

    mis_num = 0
    save_dict = {'img':[], 'degree':[]}
    for p_id, paths in enumerate(tqdm(p_d)):
        # rt to degree
        rt_path, img_path, _ = get_rt(paths, root, dataset)
        if not os.path.exists(rt_path):
            mis_num += 1
            continue
        rt = np.load(rt_path)
        degrees = []
        for rt_i in range(rt.shape[0]):
            degrees.append(rt_to_degree(rt[rt_i:rt_i+1]))
        # save
        save_dict['img'].append(img_path)
        save_dict['degree'].append(np.concatenate(degrees, axis=0))
    
    print('total miss path {}'.format(mis_num))
    # save
    with open(save_file, 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return save_dict

# sample 500 degree
def sample_degree(degrees, pickle_file):
    # read in pickle
    with open(pickle_file, 'rb') as f:
        p_d = pickle.load(f)
    end = len(p_d)
    p_id = range(0, end, end//500)
    sample_paths = [os.path.join(p_d[i][0], p_d[i][1], p_d[i][2]) for i in p_id]

    # get degree
    sam_degree = []
    for img_path, degree in zip(degrees['img'], degrees['degree']):
        if img_path in sample_paths:
            sam_degree.append(degree)
    
    return {'degree':sam_degree}

# plot degree distribution
def plot_degree(degrees, plot_path):
    degree = np.concatenate(degrees['degree'], axis=0)
    print(degree.shape)
    figure, axs = plt.subplots(nrows=3, ncols=1)
    plt.subplot(311)
    plt.hist(degree[:, 0], weights=np.ones_like(degree[:, 0])/degree.shape[0], bins=1000)
    plt.xlim(-100, 100)
    plt.title('z')
    plt.subplot(312)
    plt.hist(degree[:, 1], weights=np.ones_like(degree[:, 1])/degree.shape[0], bins=1000)
    plt.xlim(-100, 100)
    plt.title('y')
    plt.subplot(313)
    plt.hist(degree[:, 2], weights=np.ones_like(degree[:, 2])/degree.shape[0], bins=1000)
    plt.xlim(-100, 100)
    plt.title('x')
    figure.tight_layout()
    plt.savefig(plot_path)

# find example for videos in degree dict
def find_example(root, degrees, save_root):
    sels = [-75, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 75]
    sel_check = [False for i in range(len(sels)-1)]
    sel_frames = [None for i in range(len(sels)-1)]
    # check each video
    for d_id, degree in enumerate(tqdm(degrees['degree'])):
        if sum(sel_check) == len(sel_check):
            break
        video = None
        # check each degree
        for i in range(1, len(sels)):
            if sel_check[i-1]:
                continue
            check = (degree[:, 1] <= sels[i]) * (degree[:, 1] > sels[i-1])
            # exist
            if check.sum() > 1:
                sel_id = np.where(check)[0][0]
                if video is None:
                    video_path = os.path.join(root, 'test', degrees['img'][d_id]+'_crop.mp4')
                    video = read_video(video_path)
                try:
                    sel_frames[i-1] = video[sel_id]
                except:
                    pdb.set_trace()
                sel_check[i-1] = True

    # save frames
    for frame, sel in zip(sel_frames, sels[1:]):
        img_file = os.path.join(save_root, '{}.png'.format(sel))
        cv2.imwrite(img_file, frame)

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

# select certain number of videos for each bin (based on ratio)
def sel_bin_videos(save_dict, bin_list, sel_nums, shift=75, bin_len=5):
    degree_count = []
    # for each video
    for degrees in tqdm(save_dict['degree']):
        cur_count = np.zeros(len(bin_list)-1)
        # count
        for d in degrees[:, 1]+shift:
            if d > shift*2:
                d = shift*2 - 1
            if d < 0:
                d = 0

            cur_count[int(d//bin_len)] += 1
        degree_count.append(cur_count / cur_count.sum())
    # select videos
    degree_count = np.vstack(degree_count)
    degree_count_temp = degree_count.copy()
    video_paths = []
    degree_ratio = []
    for i in range(degree_count_temp.shape[1]):
        video_ids = np.argsort(degree_count_temp[:, i])[::-1]
        video_paths.append([save_dict['img'][v_id] for v_id in video_ids[:sel_nums[i]]])
        degree_ratio.append([degree_count_temp[v_id, i] for v_id in video_ids[:sel_nums[i]]])

        degree_count_temp[video_ids[:sel_nums[i]], :] = 0

    video_sel = {'path':video_paths, 'degree_ratio':degree_ratio, 'degree':bin_list-shift}
    return video_sel

# select reference and target frames for confusion matrix
def set_bin_matrix(save_dict, bin_list, save_path, sel_thre=100, shift=75, bin_len=5):
    matrix_store = {s_b:{s_b:[] for s_b in range(len(bin_list)-1)} for s_b in range(len(bin_list)-1)}
    count_store = {s_b:{s_b:0 for s_b in range(len(bin_list)-1)} for s_b in range(len(bin_list)-1)}
    degree_bins = []
    # for each video
    print('searching...')
    for degrees, video in zip(tqdm(save_dict['degree']), save_dict['img']):
        degree_bin = [-1 for i in range(degrees.shape[0])]
        # count
        for d_i, d in enumerate(degrees[:, 1]+shift):
            if d > shift*2 or d < 0:
                continue
            degree_bin[d_i] = int(d//bin_len)
        degree_bins.append(degree_bin)
        # save to matrix
        for low_id in range(len(degree_bin)-10):
            for top_id in range(low_id+10, len(degree_bin)-10):
                if degree_bin[low_id] == -1 or degree_bin[top_id] == -1:
                    continue
                count_store[degree_bin[low_id]][degree_bin[top_id]] += 1
                count_store[degree_bin[top_id]][degree_bin[low_id]] += 1

    #  get min
    min_frame = sel_thre
    for ref_id in count_store:
        min_frame = min(min_frame, min([count_store[ref_id][tgt_id] for tgt_id in count_store[ref_id]]))
    
    # print
    for top_id in count_store:
        print("{}:{}".format(top_id, count_store[top_id]))
    print('final min frame number: {}'.format(min_frame))

    # random shuffle
    random_ids = random.sample(list(range(len(degree_bins))), k=len(degree_bins))
    degree_bins = np.asarray(degree_bins)[random_ids]
    save_dict['img'] = np.asarray(save_dict['img'])[random_ids]
    save_dict['degree'] = np.asarray(save_dict['degree'])[random_ids]

    # clean
    print('cleaning...')
    count_store = {}
    for i in tqdm(range(len(degree_bins))):
        degree_bin = degree_bins[i]
        # save
        for low_id in range(len(degree_bin)-10):
            for top_id in range(low_id+10, len(degree_bin)-10):
                if degree_bin[low_id] == -1 or degree_bin[top_id] == -1:
                    continue
                if len(matrix_store[degree_bin[low_id]][degree_bin[top_id]]) < min_frame:
                    matrix_store[degree_bin[low_id]][degree_bin[top_id]].append([save_dict['img'][i], low_id, top_id])
                if len(matrix_store[degree_bin[top_id]][degree_bin[low_id]]) < min_frame:
                    matrix_store[degree_bin[top_id]][degree_bin[low_id]].append([save_dict['img'][i], top_id, low_id])

    # save
    if save_path is not None:
        with open(save_path, 'wb') as f:
            matrix_store['bin_list'] = bin_list
            pickle.dump(matrix_store, f, protocol=pickle.HIGHEST_PROTOCOL)

    return matrix_store

# find equal frame for each element in matrix
def clean_matrix_videos(matrix_dict, save_file):
    # get min frame nums
    min_frame = 500
    for ref_id in matrix_dict:
        min_frame = min(min_frame, min([len(matrix_dict[ref_id][tgt_id]) for tgt_id in matrix_dict[ref_id]]))

    # random select specific frames
    for ref_id in tqdm(matrix_dict):
        for tgt_id in matrix_dict[ref_id]:
            matrix_dict[ref_id][tgt_id] = random.choices(matrix_dict[ref_id][tgt_id], k=min_frame)

    # print
    print('finally get {} frames for each element'.format(min_frame))

    # save file
    with open(save_file, 'wb') as f:
        pickle.dump(matrix_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return matrix_dict

# get degree from rotation, select video for bin dict
def deal_with_data():
    pickle_file = 'voxceleb2/pickle/test_lmark2img.pkl'
    save_file = 'degree_store/vox/test_degree.pkl'
    root = 'voxceleb2'
    dataset = 'vox'
    save_dict = save_degree_from_rt(pickle_file, save_file, root, dataset)
    # with open(save_file, 'rb') as f:
    #     save_dict = pickle.load(f)

    plot_path = 'save/vox/sample_plot_degree.png'
    plot_degree(save_dict, plot_path)

    save_root = 'save/frames_2'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    find_example(root, save_dict, save_root)

    bin_list = np.arange(-75, 75+5, 5) + 75
    sel_nums = [20 for i in np.arange(-75, -50, 5)] + \
               [10 for i in np.arange(-50, -25, 5)] + \
               [5 for i in np.arange(-25, 25, 5)] + \
               [10 for i in np.arange(25, 50, 5)] + \
               [20 for i in np.arange(50, 75, 5)]
    video_sel = sel_bin_videos(save_dict, bin_list, sel_nums)
    save_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/vox/video_sel.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(video_sel, f, protocol=pickle.HIGHEST_PROTOCOL)

# select frames for confusion matrix
def deal_with_matrix():
    pickle_file = 'voxceleb2/pickle/test_lmark2img.pkl'
    save_file = 'degree_store/vox/test_degree.pkl'
    video_sel_file = 'degree_store/vox/video_sel.pkl'
    save_path = 'degree_store/vox/matrix_sel.pkl'
    root = 'vox'
    dataset = 'vox'
    # get dict
    with open(save_file, 'rb') as f:
        save_dict = pickle.load(f)

    # get matrix
    bin_list = np.arange(-75, 75+10, 10) + 75
    set_bin_matrix(save_dict, bin_list, save_path, bin_len=10)

    # clean dict
    clean_path = 'degree_store/vox/matrix_sel_clean.pkl'
    with open(save_path, 'rb') as f:
        video_dict = pickle.load(f)
    clean_matrix_videos(video_dict, clean_path)


if __name__ == '__main__':
    deal_with_data()
    deal_with_matrix()