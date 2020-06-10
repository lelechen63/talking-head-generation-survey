# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import numpy as np
import torch
import cv2
from collections import OrderedDict
from PIL import Image
import pickle as pkl

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

import mmcv
from tqdm import tqdm
import pdb

import warnings
warnings.simplefilter('ignore')

def get_good_file():
    pickle_data = []
    root_path = os.path.join('evaluation_store', 'good')
    # root_path = "/home/cxu-serve/p1/common/other/lrs_good"
    files = os.listdir(root_path)
    for f in files:
        pickle_data.append(f)

    # pickle_data = pickle_data[len(pickle_data)//2:]
    return pickle_data

def pick_degree_videos(dataset_name):
    # get video
    video_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/{}/video_sel.pkl'.format(dataset_name)
    with open(video_file, 'rb') as f:
        p_data = pkl.load(f)

    # get video path
    video_path = []
    for path in p_data['path']:
        video_path += path

    return list(set(video_path))

def pick_matrix_videos(dataset_name):
    # get video and frames
    video_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/{}/matrix_sel.pkl'.format(dataset_name)
    with open(video_file, 'rb') as f:
        p_data = pkl.load(f)

    # get video path
    video_ids = {}
    for ref_id in p_data:
        if ref_id == 'bin_list':
            continue
        tgt_dict = p_data[ref_id]
        # store for specific target bin
        for tgt_id in tgt_dict:
            # for each video
            for video in tgt_dict[tgt_id]:
                video_name = video[0]
                video_ref = video[1]
                video_tgt = video[2]
                if video_name not in video_ids:
                    video_ids[video_name] = {}
                if video_ref not in video_ids[video_name]:
                    video_ids[video_name][video_ref] = []
                video_ids[video_name][video_ref].append(video_tgt)

    return video_ids

def add_audio(video_name, audio_dir):
    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')

    print (command)
    os.system(command)

def image_to_video(sample_dir = None, video_name = None):
    
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.jpg -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 
    print (command)
    os.system(command)

def get_param(root, pickle_data, pick_id, opt):
    paths = pickle_data[pick_id]
    if opt.dataset_name == 'vox':
        # target
        audio_package = 'unzip/test_video'
        # audio_package = 'unzip/dev_video'
        opt.tgt_video_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned.mp4")
        if opt.no_head_motion:
            opt.tgt_lmarks_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned_front.npy")
        else:
            opt.tgt_lmarks_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned.npy")
        opt.tgt_rt_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned_ani.mp4")
        opt.tgt_audio_path = os.path.join(root, 'unzip/test_audio', paths[0], paths[1], paths[2]+".wav")

        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, audio_package, ref_paths[0], ref_paths[1], ref_paths[2]+"_aligned_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = os.path.join(root, audio_package, paths[0], paths[1], paths[2]+"_aligned.npy")
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = int(ref_paths[3])
        if opt.no_head_motion:
            opt.ref_img_id = str(opt.ref_ani_id)
            opt.n_shot = 1

        audio_tgt_path = os.path.join(root, 'unzip/test_audio', paths[0], paths[1], paths[2]+".wav")

    elif opt.dataset_name == 'grid':
        # target
        opt.tgt_video_path = os.path.join(root, 'align', paths[0], paths[1]+"_crop.mp4")
        opt.tgt_lmarks_path = os.path.join(root, 'align', paths[0], paths[1]+"_original.npy")
        opt.tgt_rt_path = os.path.join(root, 'align', paths[0], paths[1]+ '_rt.npy') 
        opt.tgt_ani_path = None
        # reference
        ref_paths = paths
        opt.ref_front_path = None
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = None

        audio_tgt_path = os.path.join(root, 'audio', paths[0], paths[1]+".wav")

    elif opt.dataset_name == 'lrs':
        # target
        paths[1] = paths[1].split('_')[0]
        opt.tgt_video_path = os.path.join(root, 'test', paths[0], paths[1]+"_crop.mp4")
        opt.tgt_lmarks_path = os.path.join(root, 'test', paths[0], paths[1]+"_original.npy")
        opt.tgt_rt_path = os.path.join(root, 'test', paths[0], paths[1]+"_rt.npy")
        opt.tgt_ani_path = os.path.join(root, 'test', paths[0], paths[1]+"_ani.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, 'test', paths[0], paths[1]+"_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        if opt.warp_ani:
            opt.ref_ani_id = int(paths[2])

        audio_tgt_path = os.path.join(root, 'test', paths[0], paths[1]+".wav")

    elif opt.dataset_name == 'crema':
        # target
        opt.tgt_video_path = os.path.join(root, 'VideoFlash', paths[0][:-10]+"_crop.mp4")
        opt.tgt_lmarks_path = os.path.join(root, 'VideoFlash', paths[0][:-10]+"_original.npy")
        opt.tgt_rt_path = os.path.join(root, 'VideoFlash', paths[0][:-10]+"_rt.npy")
        opt.tgt_ani_path = None
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, 'VideoFlash', paths[0][:-10]+"_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = None

        audio_tgt_path = os.path.join(root, 'AudioWAV', paths[0][:-11]+".wav")

    elif opt.dataset_name == 'lisa':
        # target
        opt.tgt_video_path = os.path.join(root, paths[0], paths[1], paths[2]+"_aligned.mp4")
        opt.tgt_lmarks_path = os.path.join(root, "_aligned.npy")
        opt.tgt_rt_path = os.path.join(root, 'unzip/test_video', paths[0], paths[1], paths[2]+"_aligned_rt.npy")
        opt.tgt_ani_path = os.path.join(root, 'unzip/test_video', paths[0], paths[1], paths[2]+"_aligned_ani.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(root, 'unzip/test_video', ref_paths[0], ref_paths[1], ref_paths[2]+"_aligned_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = int(ref_paths[3])

        audio_tgt_path = os.path.join(root, 'unzip/test_audio', paths[0], paths[1], paths[2]+".m4a")

    elif opt.dataset_name == 'lrw':
        # target
        opt.tgt_video_path = os.path.join(paths[0]+"_crop.mp4")
        opt.tgt_lmarks_path = os.path.join(paths[0]+"_original.npy")
        opt.tgt_rt_path = os.path.join(paths[0]+"_rt.npy")
        opt.tgt_ani_path = os.path.join(paths[0]+"_ani.mp4")
        # reference
        ref_paths = paths
        opt.ref_front_path = os.path.join(ref_paths[0]+"_front.npy")
        opt.ref_video_path = opt.tgt_video_path
        opt.ref_lmarks_path = opt.tgt_lmarks_path
        opt.ref_rt_path = opt.tgt_rt_path
        opt.ref_ani_id = int(ref_paths[1])

        audio_tgt_path = os.path.join(paths[0].replace('video', 'audio')+".wav")

    return audio_tgt_path



def change_param_for_matrix(ref_ids, tgt_ids):
    opt.ref_img_id = ','.join([str(r_i) for r_i in ref_ids])
    opt.tgt_ids = ','.join([str(t_i) for t_i in tgt_ids])

opt = TestOptions().parse()

### setup models
model = create_model(opt)
model.eval()

root = opt.dataroot
if opt.dataset_name == 'grid':
    _file = open(os.path.join(root, 'pickle','test_audio2lmark_grid.pkl'), "rb")
elif opt.dataset_name == 'crema':
    _file = open(os.path.join(root, 'pickle','train_lmark2img.pkl'), "rb")
elif opt.dataset_name == 'lrw':
    _file = open(os.path.join(root, 'pickle','test3_lmark2img.pkl'), "rb")
elif opt.dataset_name == 'lrs':
    _file = open(os.path.join(root, 'pickle','new_test_lmark2img.pkl'), "rb")
else:
    _file = open(os.path.join(root, 'pickle','test_lmark2img.pkl'), "rb")
    # _file = open(os.path.join(root, 'pickle','dev_lmark2img.pkl'), "rb")
pickle_data = pkl.load(_file)
_file.close()

if opt.dataset_name == 'crema':
    pickle_data = pickle_data[int(len(pickle_data)*0.8):]

pickle_dict = pick_matrix_videos(opt.dataset_name)
video_files = list(pickle_dict.keys())

save_name = opt.name
if opt.dataset_name == 'lrs':
    save_name = 'lrs'
if opt.dataset_name == 'lrw':
    save_name = 'lrw'

save_root = os.path.join('matrix_all_wang', save_name, '{}_shot_epoch_{}'.format(opt.n_shot, opt.which_epoch))

# pick_ids = [i for i, paths in enumerate(pickle_data) if os.path.join(paths[0], paths[1], paths[2]) in video_files]
pick_ids = [i for i, paths in enumerate(pickle_data) if os.path.join(paths[0], paths[1]) in video_files]
# pick_ids = [i for i, paths in enumerate(pickle_data) if os.path.join(paths[0]) in video_files]

# print('after {}'.format(len(pick_ids)))
pick_ids = pick_ids[:opt.how_many]

count = 0
for pick_id in tqdm(pick_ids):
    # pdb.set_trace()
    paths = pickle_data[pick_id]
    # video_name = os.path.join(paths[0], paths[1], paths[2])
    video_name = os.path.join(paths[0], paths[1])
    # video_name = os.path.join(paths[0])

    print('process {} ...'.format(pick_id))

    # for each reference image
    for ref_id in tqdm(pickle_dict[video_name]):
        change_param_for_matrix(ref_ids=[ref_id], tgt_ids=pickle_dict[video_name][ref_id])

        audio_tgt_path = get_param(root, pickle_data, pick_id, opt)

        ### setup dataset
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()

        # test
        # ref_idx_fix = torch.zeros([opt.batchSize])
        ref_idx_fix = None
        for i, data in enumerate(dataset):
            if i >= len(dataset): break
            img_path = data['path']
            if not opt.warp_ani:
                data.update({'ani_image':None, 'ani_lmark':None, 'cropped_images':None, 'cropped_lmarks':None })
            if "warping_ref" not in data:
                data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})
            # data.update({'warping_ref': data['ref_image'][:, :1], 'warping_ref_lmark': data['ref_label'][:, :1]})
            # data['tgt_audio'] = None

            img_path = data['path']
            data_list = [data['tgt_label'], data['tgt_image'], None, None, None, None, \
                        data['ref_label'], data['ref_image'], \
                        data['warping_ref_lmark'].squeeze(1) if data['warping_ref_lmark'] is not None else None, \
                        data['warping_ref'].squeeze(1) if data['warping_ref'] is not None else None, \
                        None, \
                        data['ani_lmark'].squeeze(1) if opt.warp_ani else None, \
                        data['ani_image'].squeeze(1) if opt.warp_ani else None, \
                        None, None, None, \
                        data['tgt_audio'] if opt.audio_drive and data['tgt_audio'] is not None else None, \
                        None, None]
            synthesized_image, fake_raw_img, warped_img, _, weight, _, _, _, _, _ = model(data_list, ref_idx_fix=ref_idx_fix)
            
            # save compare
            visuals = [
                util.tensor2im(data['tgt_gt_label']) if 'tgt_gt_label' in data else util.tensor2im(data['tgt_label']), \
                util.tensor2im(data['tgt_label']), \
                util.tensor2im(data['tgt_image']), \
                util.tensor2im(synthesized_image), \
                util.tensor2im(fake_raw_img), \
                util.tensor2im(warped_img[0]), \
                util.tensor2im(weight[0]), \
                util.tensor2im(warped_img[2]), \
                util.tensor2im(weight[2])
            ]
            # visuals = [util.tensor2im(synthesized_image)]
            compare_image = np.hstack([v for v in visuals if v is not None])

            img_id = "{}_{}_{}".format(img_path[0].split('/')[-3], img_path[0].split('/')[-2], img_path[0].split('/')[-1][:-4])
            img_dir = os.path.join(save_root,  img_id, str(ref_id))
            img_name = "%05d.jpg"%data['index'][0]

            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            image_pil = Image.fromarray(compare_image)
            image_pil.save(os.path.join(img_dir, img_name))

            # save reference
            if i == 0:
                if not os.path.exists(os.path.join(img_dir, 'reference')):
                    os.makedirs(os.path.join(img_dir, 'reference'))
                for ref_img_id in range(data['ref_image'].shape[1]):
                    ref_img = util.tensor2im(data['ref_image'][0, ref_img_id])
                    ref_img = Image.fromarray(ref_img)
                    ref_img.save(os.path.join(img_dir, 'reference', 'ref_{}.png').format(ref_img_id))
