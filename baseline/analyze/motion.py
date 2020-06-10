import os
import matplotlib.pyplot as plt
import pickle
import pdb
import random
import numpy as np

# select video for motion
def sel_bin_motion(save_dict, sel_dict, save_path, bin_list, sel_video=10):
    bin_len = bin_list[1] - bin_list[0]
    bin_max = max(bin_list)

    # map video and motion
    sel_motion_result = [max(degree[:,1]) - min(degree[:,1]) for degree in sel_dict['degree']]
    bin_dict = {motion_id:[] for motion_id in range(len(bin_list)-1)}
    for motion, video in zip(sel_motion_result, sel_dict['img']):
        if motion >= bin_max:
            motion = bin_max - 1
        bin_dict[int(motion // bin_len)].append(video)

    # get total video
    motion_result = [max(degree[:,1]) - min(degree[:,1]) for degree in save_dict['degree']]
    total_bin_dict = {motion_id:[] for motion_id in range(len(bin_list)-1)}
    for motion, video in zip(motion_result, save_dict['img']):
        if motion >= bin_max:
            continue
        total_bin_dict[int(motion // bin_len)].append(video)
    
    # select video
    video_dict = {motion_id:[] for motion_id in range(len(bin_list)-1)}
    extra_video_dict = {motion_id:[] for motion_id in range(len(bin_list)-1)}
    for motion_id in video_dict:
        if len(bin_dict[motion_id]) >= sel_video:
            video_dict[motion_id].extend(random.sample(bin_dict[motion_id], k=sel_video))
        else:
            video_dict[motion_id] += bin_dict[motion_id]
            # update extract
            try:
                extra_video_list = random.sample(total_bin_dict[motion_id], k=sel_video)
            except:
                pdb.set_trace()
            extra_video_list = [video for video in extra_video_list \
                                  if video not in video_dict[motion_id]][:sel_video-len(video_dict[motion_id])]
            extra_video_dict[motion_id] += extra_video_list
            # update total video
            video_dict[motion_id] += extra_video_list

    # save dict
    with open(save_path, 'wb') as f:
        pickle.dump([video_dict, extra_video_dict, bin_list], f, protocol=pickle.HIGHEST_PROTOCOL)

    return [video_dict, extra_video_dict, bin_list]

def plot_motion_distribution(save_dict, fig_path, extract=None):
    # video as one
    if extract is None:
        motion_list = [max(degree[:,1]) - min(degree[:,1]) for degree in save_dict['degree']]
    # extract frames from video
    else:
        motion_list = []
        for degree in save_dict['degree']:
            for frame_degree in range(0, len(degree), extract):
                cur_degree = degree[frame_degree:frame_degree+extract]
                motion_list.append(max(cur_degree[:,1]) - min(cur_degree[:,1]))
    print('total frames {}'.format(len(motion_list)))
    plt.figure()
    plt.hist(motion_list, bins=100, weights=np.ones(len(motion_list))/len(motion_list))
    plt.savefig(fig_path)
    plt.close()

# plot motion distribution and select video for motion
def deal_with_data():
    degree_file = 'degree_store/vox/test_degree.pkl'
    sel_file = 'degree_store/vox/video_sel.pkl'
    # get degree dict
    fig_path = 'move_store/vox/motion_figures/plot.png'
    with open(degree_file, 'rb') as f:
        degree_dict = pickle.load(f)
    with open(sel_file, 'rb') as f:
        sel_dict = pickle.load(f)
        video_sels = []
        for video_list in sel_dict['path']:
            video_sels += video_list
        sel_dict = {}
    degree_sel_dict = {'degree':[], 'img':[]}
    for video, degree in zip(degree_dict['img'], degree_dict['degree']):
        if video in video_sels:
            degree_sel_dict['degree'].append(degree)
            degree_sel_dict['img'].append(video)

    # plot
    plot_motion_distribution(degree_dict, fig_path)
    
    # save selected video
    save_root = 'move_store/vox'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    bin_list = np.arange(0, 120+10, 10)
    results = sel_bin_motion(degree_dict, degree_sel_dict, os.path.join(save_root, 'video_sel.pkl'), bin_list, sel_video=9)

if __name__ == '__main__':
    deal_with_data()