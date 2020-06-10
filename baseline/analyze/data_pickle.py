import os
import pickle
import pdb
import cv2
import random
import pandas as pd
from tqdm import tqdm
import numpy as np


def collect_pickle(test_root, save_file):
    pickle_paths = []
    id_paths = os.listdir(test_root)
    for id_p in id_paths:
        cur_path = os.path.join(test_root, id_p)
        video_paths = []
        cur_files = os.listdir(cur_path)
        for file in cur_files:
            if file[6:]=='crop.mp4' and file[:5]+'_rt.npy' in cur_files and file[:5]+'_original.npy' in cur_files:
                video_paths.append(file[:5])
        pickle_paths.extend([id_p, v_p] for v_p in video_paths)

    print('pickle len is {}'.format(len(pickle_paths)))

    with open(save_file, 'wb') as f:
        pickle.dump(pickle_paths, f, protocol=pickle.HIGHEST_PROTOCOL)

def collect_pickle_face(save_file):
    pickle_paths = []
    for i in range(1000):
        pickle_paths.append(['%03d.mp4'%i, '%03d_rt.npy'%i])

    print('pickle len is {}'.format(len(pickle_paths)))

    with open(save_file, 'wb') as f:
        pickle.dump(pickle_paths, f, protocol=pickle.HIGHEST_PROTOCOL)

def collect_pickle_obama(save_file):
    root_path = '/home/cxu-serve/p1/common/Obama/video'
    files = os.listdir(root_path)
    pickle_paths = [file[:-9] for file in files if file[-7:]=='rt2.npy']

    print('pickle len is {}'.format(len(pickle_paths)))

    with open(save_file, 'wb') as f:
        pickle.dump(pickle_paths, f, protocol=pickle.HIGHEST_PROTOCOL)

def collect_csv_newlrs(save_file, save_dir):
    np_path = '/home/cxu-serve/p1/common/lrs3_lip.npy'
    files = np.load(np_path)
    csv_dict = {'ref_path':[], 'audio_path':[], 'start':[], 'end':[]}
    # get csv
    for path_id, path in enumerate(tqdm(files)):
        file, start_index, end_index, _ = path
        video = read_video(file)
        save_part = '_'.join(file.split('/')[-2:])
        fig_file = os.path.join(save_dir, save_part[:-4]+'_0.png')
        audio_file = file[:-9]+'.wav'
        landmark_file = file[:-9]+'_original.npy'
        rt_file = file[:-9]+'_rt.npy'
        if not os.path.exists(audio_file) or not os.path.exists(landmark_file) or not os.path.exists(rt_file):
            print('miss')
            continue
        cv2.imwrite(fig_file, video[0])
        csv_dict['ref_path'].append(fig_file)
        csv_dict['audio_path'].append(audio_file)
        csv_dict['start'].append(start_index)
        csv_dict['end'].append(end_index)
    # store
    print('valid file: {}'.format(len(csv_dict['ref_path'])))
    pd.DataFrame.from_dict(csv_dict).to_csv(save_file, index=False)

def collect_csv_newvox(save_file, save_dir):
    np_path = '/home/cxu-serve/p1/common/vox_lip.npy'
    files = np.load(np_path)
    csv_dict = {'ref_path':[], 'audio_path':[], 'start':[], 'end':[]}
    # get csv
    for path_id, path in enumerate(tqdm(files)):
        file, start_index, end_index, _ = path
        video = read_video(file)
        save_part = '_'.join(file.split('/')[-3:])
        fig_file = os.path.join(save_dir, save_part[:-4]+'_0.png')
        audio_file = file.replace('video', 'audio').replace('_aligned.mp4', '.wav')
        landmark_file = file.replace('_aligned.mp4', '_aligned.npy')
        rt_file = file.replace('_aligned.mp4', '_aligned_rt.npy')
        if not os.path.exists(audio_file) or not os.path.exists(landmark_file) or not os.path.exists(rt_file):
            print('miss')
            continue
        cv2.imwrite(fig_file, video[0])
        csv_dict['ref_path'].append(fig_file)
        csv_dict['audio_path'].append(audio_file)
        csv_dict['start'].append(start_index)
        csv_dict['end'].append(end_index)
    # store
    print('valid file: {}'.format(len(csv_dict['ref_path'])))
    pd.DataFrame.from_dict(csv_dict).to_csv(save_file, index=False)

def get_finetune_pickle(pickle_file, del_file, save_file):
    # load in pickle
    with open(pickle_file, 'rb') as f:
        p_data = pickle.load(f)
    # load in delete
    with open(del_file, 'rb') as f:
        del_data = pickle.load(f)
    # get video
    del_videos = []
    for video in del_data['path']:
        del_videos += video
    # get finetune pickle
    # finetune_data = [paths for paths in p_data if os.path.join(paths[0], paths[1]) not in del_videos]
    finetune_data = [paths for paths in p_data if os.path.join(paths[0]) not in del_videos]

    print('finetune pickle len is {}'.format(len(finetune_data)))

    with open(save_file, 'wb') as f:
        pickle.dump(finetune_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def pick_grid_pickle(pickle_file, save_file, sel_num=10):
    # load in pickle
    with open(pickle_file, 'rb') as f:
        p_data = pickle.load(f)
    # store dict
    persons = []
    data_dict = {}
    for paths in p_data:
        person_id = paths[0]
        video_id = paths[1]
        persons.append(person_id)
        if person_id not in data_dict:
            data_dict[person_id] = []
        data_dict[person_id].append(video_id)
    # select randomly and new pickle
    new_pickle = []
    for person_id in data_dict:
        video_ids = random.sample(data_dict[person_id], k=sel_num)
        for video_id in video_ids:
            new_pickle.append([person_id, video_id])
    # save pickle
    print('total datas {}'.format(len(new_pickle)))
    with open(save_file, 'wb') as f:
        pickle.dump(new_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)

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

def pickle_to_csv(pickle_file, csv_file, img_root):
    # load in pickle
    with open(pickle_file, 'rb') as f:
        p_data = pickle.load(f)
    # get video and audio
    p_dict = {'img':[], 'audio':[]}
    for paths in tqdm(p_data):
        # video_path = os.path.join('/home/cxu-serve/p1/common/grid', 'align', paths[0], paths[1]+"_crop.mp4")
        # audio_path = os.path.join('/home/cxu-serve/p1/common/grid', 'audio', paths[0], paths[1]+".wav")
        # frame_path = os.path.join(img_root, '{}_{}'.format(paths[0], paths[1]+'_0.png'))

        video_path = os.path.join('/home/cxu-serve/p1/common/voxceleb2', 'unzip/test_video', \
            paths[0], paths[1], paths[2]+"_crop.mp4")
        audio_path = os.path.join('/home/cxu-serve/p1/common/voxceleb2', 'unzip/test_audio', \
            paths[0], paths[1], paths[2]+".wav")
        frame_path = os.path.join(img_root, '{}_{}'.format(paths[0], paths[1]+'_0.png'))

        landmark_path = os.path.join('/home/cxu-serve/p1/common/voxceleb2', 'unzip/test_video', \
            paths[0], paths[1], paths[2]+"_aligned.npy")
        rt_path = os.path.join('/home/cxu-serve/p1/common/voxceleb2', 'unzip/test_video', \
            paths[0], paths[1], paths[2]+"_aligned_rt.npy")

        if not os.path.exists(audio_path) or not os.path.exists(landmark_path) or not os.path.exists(rt_path):
            print('miss')
            pdb.set_trace()
            continue 

        # real_video = read_video(video_path)
        # frame = real_video[0]
        # cv2.imwrite(frame_path, frame)

        p_dict['img'].append(frame_path)
        p_dict['audio'].append(audio_path)
    
    # save file
    pd.DataFrame.from_dict(p_dict).to_csv(csv_file, index=False)

def files_to_csv(video_dir, csv_file, img_root):
    # load in pickle
    video_file = os.listdir(video_dir)
    video_file = [video[17:] for video in video_file]
    
    # get video and audio
    p_dict = {'img':[], 'audio':[]}
    for video in tqdm(video_file):
        video_path = os.path.join('/home/cxu-serve/p1/common/CREMA/VideoFlash', video+".mp4")
        audio_path = os.path.join('/home/cxu-serve/p1/common/CREMA/AudioWAV', video+".wav")
        frame_path = os.path.join(img_root, video+'_0.png')

        try:
            real_video = read_video(video_path)
            frame = real_video[0]
            cv2.imwrite(frame_path, frame)

            p_dict['img'].append(frame_path)
            p_dict['audio'].append(audio_path)
        except:
            print('wrong path {}'.format(video_path))

    # save file
    pd.DataFrame.from_dict(p_dict).to_csv(csv_file, index=False)

def dir_data_to_pickle_grid(fake_root, save_file):
    save_list = []
    root = '/home/cxu-serve/p1/common/grid'
    for video_name in tqdm(os.listdir(fake_root)):
        video_split = video_name.split('_')
        video = video_split[:2]+['_'.join(video_split[2:])]
        video = '/'.join(video)+'.mp4'
        video = os.path.join(root, video)
        assert os.path.exists(video)
        save_list.append(video)

    # store
    print('totally {} videos'.format(len(save_list)))
    with open(save_file, 'wb') as f:
        pickle.dump(save_list, f)

if __name__ == '__main__':
    # test_root = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/test'
    # save_file = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pickle/new_test_lmark2img.pkl'
    # collect_pickle(test_root, save_file)

    # pickle_file = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pickle/new_test_lmark2img.pkl'
    # del_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/lrs/video_sel.pkl'
    # save_file = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pickle/new_finetune_lmark2img.pkl'
    # get_finetune_pickle(pickle_file, del_file, save_file)

    # pickle_file = '/home/cxu-serve/p1/common/lrw/pickle/test3_lmark2img.pkl'
    # del_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/lrw/video_sel.pkl'
    # save_file = '/home/cxu-serve/p1/common/lrw/new_pickle/new_finetune_lmark2img.pkl'
    # get_finetune_pickle(pickle_file, del_file, save_file)

    # pickle_file = '/home/cxu-serve/p1/common/grid/pickle/degree_test.pkl'
    # csv_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/grid/sample_grid.csv'
    # img_root = '/home/cxu-serve/p1/common/degree_frames/grid_degree/degree_frames'
    # pickle_to_csv(pickle_file, csv_file, img_root)

    # pickle_file = '/home/cxu-serve/p1/common/voxceleb2/pickle/test_lmark2img.pkl'
    # csv_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/vox/test_total.csv'
    # img_root = '/home/cxu-serve/p1/common/degree_frames/vox_degree/degree_frames'
    # pickle_to_csv(pickle_file, csv_file, img_root)

    # pickle_file = '/home/cxu-serve/p1/common/voxceleb2/pickle/test_lmark2img.pkl'
    # csv_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/vox/test_total.csv'
    # img_root = '/home/cxu-serve/p1/common/degree_frames/vox_degree/degree_frames'
    # pickle_to_csv(pickle_file, csv_file, img_root)

    # video_file = '/home/cxu-serve/p1/common/degree/crema_results/baseline/epoch_20'
    # csv_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/CREMA/crema.csv'
    # img_root = '/home/cxu-serve/p1/common/degree_frames/crema_degree/degree_frames'
    # files_to_csv(video_file, csv_file, img_root)

    # save_file = '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/degree_pickle/test.pkl'
    # collect_pickle_face(save_file)

    # save_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/obama/total_data.pkl'
    # collect_pickle_obama(save_file)

    # pickle_file = '/home/cxu-serve/p1/common/grid/pickle/test_audio2lmark_grid.pkl'
    # save_file = '/home/cxu-serve/p1/common/grid/pickle/degree_test.pkl'
    # pick_grid_pickle(pickle_file, save_file, 10)

    # save_file =  '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/lrs/new_extra_data.csv'
    # img_root = '/home/cxu-serve/p1/common/degree_frames/lrs_degree/degree_new_frame'
    # collect_csv_newlrs(save_file, img_root)

    # save_file =  '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/vox/new_extra_data.csv'
    # img_root = '/home/cxu-serve/p1/common/degree_frames/vox_degree/degree_new_frame'
    # if not os.path.exists(img_root):
    #     os.makedirs(img_root)
    # collect_csv_newvox(save_file, img_root)

    fake_root = '/home/cxu-serve/p1/common/degree/grid_results/baseline'
    save_file = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/grid/blink_need.pkl'
    dir_data_to_pickle_grid(fake_root, save_file)