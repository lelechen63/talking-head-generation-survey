# Load the dataset
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
from torchvision.transforms import ToTensor, Scale, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Resize
from PIL import Image
import cv2

import pdb

import os

class Grid(data.Dataset):
    def __init__(self, files, nums):
        super(Grid, self).__init__()
        self.files = files
        self.nums = nums

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        imgs = np.sort(os.listdir(path))
        img_nums = [int(img_id * len(imgs) / self.nums) for img_id in range(self.nums)]

        img_list = [self.load_imgs(os.path.join(path, imgs[num])) for num in img_nums]

        return index, img_list

    def load_imgs(self, img_file):
        img = Image.open(img_file)
        transform = Compose([Scale((256,256)), ToTensor()])
        return Variable(transform(img)).cuda()

    def get_file(self, index):
        return self.files[index]

class Vox(data.Dataset):
    def __init__(self, files, nums):
        super(Vox, self).__init__()
        self.files = files
        self.nums = nums

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        imgs = self.read_videos(path)
        img_nums = [int(img_id * len(imgs) / self.nums) for img_id in range(self.nums)]
        img_list = [self.load_imgs(imgs[num]) for num in img_nums]

        return index, img_list

    def load_imgs(self, img):
        img_result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = Compose([Scale((256,256)), ToTensor()])
        return Variable(transform(img_result)).cuda()

    def read_videos(self, video_path):
        cap = cv2.VideoCapture(video_path)
        real_video = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                real_video.append(frame)
            else:
                break

        return real_video

    def get_file(self, index):
        return self.files[index]

class VoxSingle(data.DataLoader):
    def __init__(self, files, ref_indx = '0,10,20,30,40,50,60,70,80'):
        # super(VoxSingle, self).__init__()
        self.files = files
        self.ref_indx = ref_indx

        self.cur_indx = 0
        self.cur_file = self.files[self.cur_indx]
        if self.cur_file.split('.')[-1] == 'mp4':
            self.imgs = self.read_videos(self.cur_file)
        elif self.cur_file.split('.')[-1] in ['png', 'jpg']:
            self.imgs = self.read_img(self.cur_file)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.load_imgs(img)
        
        return index, img

    def load_imgs(self, img):
        img_result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = Compose([Scale((256,256)), ToTensor()])
        return Variable(transform(img_result))

    def read_videos(self, video_path):
        cap = cv2.VideoCapture(video_path)
        real_video = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                real_video.append(frame)
            else:
                break

        return real_video

    def read_img(self, img_path):
        img = cv2.imread(img_path)
        return [img]

    def nextfile(self):
        self.cur_indx += 1
        if self.cur_indx >= len(self.files):
            return -1
        else:
            self.cur_file = self.files[self.cur_indx]
            self.imgs = self.read_videos(self.cur_file)
            return self.cur_indx

    def get_ref(self):
        ref_imgs = []
        ref_index = [int(ref_id) for ref_id in self.ref_indx.split(',')]
        if max(ref_index) >= len(self.imgs):
            # ref_index = list(range(0, len(self.imgs), len(self.imgs)//len(ref_index)))[:len(ref_index)]
            ref_index = [0,2,4,6,8,10,12,14]
            print('new reference: {}'.format(ref_index))
        for ref_num in ref_index:
            ref_imgs.append(self.load_imgs(self.imgs[int(ref_num)]))
        return ref_imgs