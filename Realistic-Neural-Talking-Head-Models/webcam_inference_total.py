"""Main"""
import torch
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image

from dataset.video_extraction_conversion import select_frames, select_images_frames, generate_cropped_landmarks
from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from webcam_demo.conversion_self import *
from finetuning_training_block import finetune

import numpy as np
import copy
import pickle as pkl

from tqdm import tqdm

device = torch.device("cuda:0")
cpu = torch.device("cpu")

def add_audio(video_name, audio_dir):
    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')

    print (command)
    os.system(command)

def image_to_video(sample_dir = None, video_name = None):
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%05d.jpg -c:v libx264 -y -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"  ' + video_name 

    print (command)
    os.system(command)

def draw_fig(x, x_hat, g_y, save_root, indx):
    # plt.clf()
    out1 = x_hat.transpose(1,3)[0]/255
    out1 = out1.to(cpu).numpy()
    out2 = x.transpose(1,3)[0]/255
    out2 = out2.to(cpu).numpy()
    out3 = g_y.transpose(1,3)[0]/255
    out3 = out3.to(cpu).numpy()

    out = np.hstack([out1, out2, out3])
    # out = out1
    out = (out * 255).astype(np.uint8)

    img_pil = Image.fromarray(out)
    img_pil.save(os.path.join(save_root, '%05d.jpg' % indx))

# embed video
def embed(path_to_video, lmarks, path_to_chkpt=None, ckpt=None, T=1, E=None, ref_ids=None):
    print("embedding...")

    """Loading Embedder input"""
    if path_to_video.split('.')[-1] in ['png', 'jpg']:
        ref_img = cv2.imread(path_to_video)
        frame_mark_video = [cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)]
        lmarks_list = [lmarks.copy()]
    elif path_to_video.split('.')[-1] in ['mp4']:
        frame_mark_video, lmarks_list = select_frames(path_to_video , T, lmarks=lmarks.copy(), specific_frames=ref_ids)
    inference_img = frame_mark_video
    reference_lmarks = lmarks_list

    frame_mark_video = generate_cropped_landmarks(frame_mark_video, pad=50)
    frame_mark_video = torch.from_numpy(np.array(frame_mark_video)).type(dtype = torch.float) #T,2,256,256,3
    frame_mark_video = frame_mark_video.transpose(2,4).to(device) #T,2,3,256,256
    f_lm_video = frame_mark_video.unsqueeze(0) #1,T,2,3,256,256

    if E is None:
        E = Embedder(256).to(device)
        E.eval()

        """Loading from past checkpoint"""
        if path_to_chkpt is not None:
            checkpoint = torch.load(path_to_chkpt, map_location=cpu)
        else:
            assert ckpt is not None
            checkpoint = copy.deepcopy(ckpt)
        E.load_state_dict(checkpoint['E_state_dict'])
    else:
        E = E
        E.eval()

    """Inference"""
    with torch.no_grad():
        # Calculate average encoding vector for video
        f_lm = f_lm_video
        f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxT,2,3,224,224
        e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxT,512,1
        e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,T,512,1
        e_hat_video = e_vectors.mean(dim=1)

    return e_hat_video, inference_img, reference_lmarks

# finetune
def finetune_train(ref_imgs, ref_lmarks, path_to_chkpt=None, path_to_embedding=None, e_hat_video=None, ckpt=None):
    return finetune(ref_imgs, ref_lmarks, path_to_chkpt, path_to_embedding, path_to_save=None, e_hat=e_hat_video, num_epochs=120, ckpt=ckpt)

# synthesize
def generate(e_hat_video, path_to_video, path_to_model_weights=None, generator=None, save_root='results/', lmark=None, tgt_ids=None):
    print("generating...")

    e_hat = e_hat_video

    G = Generator(256, finetuning=True, e_finetuning=e_hat)
    G.eval()

    """Training Init"""
    if path_to_model_weights is not None:
        checkpoint = torch.load(path_to_model_weights, map_location=cpu) 
        G.load_state_dict(checkpoint['G_state_dict'])
    else:
        assert generator is not None
        G.load_state_dict(generator.state_dict())
    
    G.to(device)
    G.finetuning_init()

    """Main"""
    cap = cv2.VideoCapture(path_to_video)

    with torch.no_grad():
        success, frame = cap.read()
        i = 0
        while success:
            # if specific target ids
            if tgt_ids is not None and i not in tgt_ids:
                i += 1
                success, frame = cap.read()
                continue

            print('finish {}'.format(i))
            x, g_y = generate_landmarks(frame=frame, device=device, pad=50, lmark=lmark[i])

            g_y = g_y.unsqueeze(0)
            x = x.unsqueeze(0)

            x_hat = G(g_y, e_hat)

            draw_fig(x, x_hat, g_y, save_root, i)

            # new frame
            i += 1
            success, frame = cap.read()

    cap.release()

def test_demo():
    """Hyperparameters and config"""
    path_to_chkpt = 'checkpoints/model_weights.tar'
    dataset_name = 'vox'

    pick_data = np.load('vox_demo.npy')

    frame_dict = {}
    for f in pick_data:
        if f[0] not in frame_dict:
            frame_dict[f[0]] = []
        frame_dict[f[0]] += np.arange(int(f[1]), int(f[2])).tolist()

    path_to_ref_videos = list(set([f[0] for f in pick_data]))
    path_to_ref_videos = sorted(path_to_ref_videos)
    path_to_ref_lmarks = [f.replace('aligned.mp4', 'aligned.npy') for f in path_to_ref_videos]

    print('total {} video'.format(len(path_to_ref_lmarks)))
    path_to_tgt_videos = path_to_ref_videos
    path_to_tgt_lmarks = path_to_ref_lmarks

    T = 8
    ref_ids = [0,5,10,15,20,25,30,35]
    E = Embedder(256).to(device)
    E.eval()
    """Loading from past checkpoint"""
    ckpt = torch.load(path_to_chkpt, map_location=cpu)
    E.load_state_dict(ckpt['E_state_dict'])
    
    # synthesize
    for path_to_ref_video, path_to_ref_lmark, path_to_tgt_video, path_to_tgt_lmark in zip(tqdm(path_to_ref_videos), path_to_ref_lmarks, path_to_tgt_videos, path_to_tgt_lmarks):
        print('current {}'.format(path_to_tgt_video))
        ref_lmark = np.load(path_to_ref_lmark)
        tgt_lmark = np.load(path_to_tgt_lmark)
        img_id = '{}'.format(path_to_tgt_video.split('/')[-1])
        save_root = 'extra_degree_results/{}/{}'.format(dataset_name, img_id[:-4])

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        if os.path.exists(os.path.join(save_root, '%05d.jpg' % frame_dict[path_to_ref_video][0])):
            continue

        ################### embed video ##############################
        e_hat_video, reference_img, reference_lmarks = embed(path_to_ref_video, ref_lmark, ckpt=copy.deepcopy(ckpt), T=T, E=E, ref_ids=ref_ids)
        inf_root = os.path.join(save_root, 'inference')
        if not os.path.exists(inf_root):
            os.makedirs(inf_root)
        for img_id, img in enumerate(reference_img):
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(inf_root, '%05d.jpg'%img_id))

        ################# finetune ####################################
        G = finetune_train(reference_img, reference_lmarks, e_hat_video=e_hat_video, ckpt=copy.deepcopy(ckpt))

        ################ synthesize ###################################
        tgt_ids = frame_dict[path_to_ref_video]
        generate(e_hat_video, path_to_tgt_video, generator=G, save_root=save_root, lmark=tgt_lmark, tgt_ids=tgt_ids)
        # image_to_video(save_root, os.path.join(save_root, 'test.mp4'))


if __name__ == '__main__':
    test_demo()