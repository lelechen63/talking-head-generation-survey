import cv2
import random
import face_alignment
from matplotlib import pyplot as plt
import numpy as np
import os

from webcam_demo.webcam_extraction_conversion import crop_and_reshape_preds, crop_and_reshape_img

import pdb

def select_frames(video_path, K, lmarks=None, specific_frames=None):
    cap = cv2.VideoCapture(video_path)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #unused
    #w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # specific frame for reference video
    if specific_frames is not None:
        assert len(specific_frames) == K
        rand_frames_idx = [0]*n_frames
        if max(specific_frames) >= n_frames:
            specific_frames = [0,2,4,6,8,10,12,14]
            print('reset reference to {}'.format(specific_frames))
        for frame_id in specific_frames:
            rand_frames_idx[frame_id] = 1
    # There are not enough frames in the video
    elif n_frames <= K:
        rand_frames_idx = [1]*n_frames
    # randomly select
    else:
        rand_frames_idx = [0]*n_frames
        i = 0
        while(i < K):
            idx = random.randint(0, n_frames-1)
            if rand_frames_idx[idx] == 0:
                rand_frames_idx[idx] = 1
                i += 1
    
    frames_list = []
    if lmarks is not None:
        lmarks_list = []
    
    # Read until video is completed or no frames needed
    ret = True
    frame_idx = 0
    while(ret and frame_idx < n_frames):
        ret, frame = cap.read()
        
        if ret and rand_frames_idx[frame_idx] == 1:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(RGB)
            if lmarks is not None:
                lmarks_list.append(lmarks[frame_idx])
            
        frame_idx += 1

    cap.release()
    
    if lmarks is not None:
        return frames_list, lmarks_list
    return frames_list

def generate_landmarks(frames_list):
    frame_landmark_list = []
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')
    
    for i in range(len(frames_list)):
        try:
            input = frames_list[i]
            preds = fa.get_landmarks(input)[0]

            dpi = 100
            fig = plt.figure(figsize=(input.shape[1]/dpi, input.shape[0]/dpi), dpi = dpi)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(np.ones(input.shape))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            #chin
            ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
            #left and right eyebrow
            ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            #nose
            ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            #left and right eye
            ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            #outer and inner lip
            ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
            ax.axis('off')

            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frame_landmark_list.append((input, data))
            plt.close(fig)
        except:
            print('Error: Video corrupted or no landmarks visible')
    
    for i in range(len(frames_list) - len(frame_landmark_list)):
        #filling frame_landmark_list in case of error
        frame_landmark_list.append(frame_landmark_list[i])
    
    
    return frame_landmark_list


def select_images_frames(path_to_images):
    images_list = []
    for image_name in os.listdir(path_to_images):
        img = cv2.imread(os.path.join(path_to_images, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)
    return images_list

def generate_cropped_landmarks(frames_list, lmark_list=None, pad=50):
    frame_landmark_list = []
    if lmark_list is None:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')
    
    for i in range(len(frames_list)):
        try:
            input = frames_list[i]
            if lmark_list is None:
                preds = fa.get_landmarks(input)[0]
            else:
                preds = lmark_list[i][:-1]

            input = crop_and_reshape_img(input, preds, pad=pad)
            preds = crop_and_reshape_preds(preds, pad=pad)

            dpi = 100
            fig = plt.figure(figsize=(input.shape[1]/dpi, input.shape[0]/dpi), dpi = dpi)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(np.ones(input.shape))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            #chin
            ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
            #left and right eyebrow
            ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            #nose
            ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            #left and right eye
            ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            #outer and inner lip
            ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
            ax.axis('off')

            fig.canvas.draw()
    
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frame_landmark_list.append((input, data))
            plt.close(fig)
        except:
            print('Error: Video corrupted or no landmarks visible')
    
    for i in range(len(frames_list) - len(frame_landmark_list)):
        #filling frame_landmark_list in case of error
        frame_landmark_list.append(frame_landmark_list[i])
    
    
    return frame_landmark_list