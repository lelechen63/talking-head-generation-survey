import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torchvision.transforms as transforms
import pdb
import cv2
import pandas as pd
from tqdm import tqdm
import random
import math
import shutil
import textgrids
from scipy.io import wavfile
from collections import Counter

import torch

import sys
sys.path.append('../')
import utils


class LRS3Classification:
	def __init__(self, opt, mode='train', data_csv=None):
		self.opt = opt
		self.mode = mode
		self.data_csv = pd.read_csv('./repo/lrs3_word.csv')
		self.labels = sorted(list(set(self.data_csv['word'])))

		if mode == 'train':
			self.cur_csv = self.data_csv[self.data_csv['mode'] == 0]
		elif mode == 'val':
			self.cur_csv = self.data_csv[self.data_csv['mode'] == 1]
		else:
			self.cur_csv = self.data_csv[self.data_csv['mode'] == 2]

		self.cur_csv.reset_index(drop=True, inplace=True)
		self.t = self.trans()

	def __getitem__(self, index):
		item = self.cur_csv.loc[index]
		path = item['path']
		align = item['word']
		start = int(item['start'])
		end = int(item['end'])

		subset = path.split('/')[0]
		output_dir, filename = '/'.join(path.replace(subset,
													 subset+'_crop').split('/')[:-1]), path.split('/')[-1]
		video_vec = np.load(os.path.join(
			self.opt.lrs3_root, output_dir, filename[:-4]+'.npz'))['data']

		duration = end - start
		start -= (self.opt.min_frame - duration)//2
		end += self.opt.min_frame - (end-start)
		start = max(0, start)
		end = min(len(video_vec), end)
		video_vec = video_vec[start:end]
		if len(video_vec) < self.opt.min_frame:
			video_vec = np.concatenate([video_vec, np.zeros(
				[self.opt.min_frame-len(video_vec)]+list(video_vec.shape[1:]), np.uint8)], 0)

		video_vec = torch.stack(
			[self.t(Image.fromarray(v).convert('L')) for v in video_vec], 0)
		return video_vec, self.labels.index(align), path

	def __len__(self):
		return len(self.cur_csv)

	def trans(self):
		if self.mode == 'train':
			return transforms.Compose([
				transforms.Resize((96, 96)),
				transforms.RandomCrop((88, 88)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
			])
		else:
			return transforms.Compose([
				transforms.Resize((96, 96)),
				transforms.TenCrop((88, 88)),
				transforms.Lambda(lambda crops: torch.stack(
					[transforms.ToTensor()(crop) for crop in crops])),
			])


if __name__ == '__main__':
	pass
