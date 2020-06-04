import os
import cv2
import math
import shutil
import requests
import numpy as np
import torch


def get_imgs_from_video(video, ext='jpg', RGB=False):
	frames = []
	if os.path.isdir(video):
		frames = sorted(glob.glob(os.path.join(video, '*.{}'.format(ext))),
						key=lambda x: int(x.split('/')[-1].split('.')[0]))
		frames = [cv2.imread(f) for f in frames]
	else:
		cap = cv2.VideoCapture(video)
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			frames.append(frame)

	frames = np.array(frames)
	if RGB:
		return frames[..., ::-1]
	else:
		return frames


def get_dataset(options):
	if options.dataset == 'lrs3':
		from data.lrs3 import LRS3Classification
		dataset_train = LRS3Classification(options, mode='train')
		dataset_val = LRS3Classification(options, mode='val')
		dataset_test = LRS3Classification(options, mode='test')
	else:
		raise

	return (dataset_train, dataset_val, dataset_test)


def init_log_dir(opt):
	if os.path.exists(os.path.join('./save', opt.name)):
		print('dir exist, delete?')
		x = input()
		if x == 'y':
			shutil.rmtree(os.path.join('./save', opt.name))
		else:
			raise

	os.mkdir(os.path.join('./save', opt.name))
	with open(os.path.join('./save', opt.name, 'option.txt'), "a") as f:
		for k, v in vars(opt).items():
			f.write('{} -> {}\n'.format(k, v))

	os.mkdir(os.path.join('./save', opt.name, 'check'))
	os.mkdir(os.path.join('./save', opt.name, 'img'))
	os.mkdir(os.path.join('./save', opt.name, 'tb'))


def get_model(options):
	if options.encode == '233':
		from model.encode import D3Conv_Res34_233
		encode = D3Conv_Res34_233(options)
	else:
		encode = None

	if options.middle == 'tc':
		from model.middle import MidTemporalConv
		mid_net = MidTemporalConv(options)
	else:
		mid_net = None

	if options.decode == 'pass':
		from model.middle import Pass
		decode = Pass(options)
	else:
		decode = None

	return encode, mid_net, decode


class AdjustLR(object):
	def __init__(self, optimizer, init_lr, sleep_epochs=3, half=5):
		super(AdjustLR, self).__init__()
		self.optimizer = optimizer
		self.sleep_epochs = sleep_epochs
		self.half = half
		self.init_lr = init_lr

	def step(self, epoch):
		if epoch >= self.sleep_epochs:
			for idx, param_group in enumerate(self.optimizer.param_groups):
				new_lr = self.init_lr[idx] * \
					math.pow(0.5, (epoch-self.sleep_epochs+1)/float(self.half))
				param_group['lr'] = new_lr


def one_hot(opt, indices, depth):
	encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
	if opt.gpu:
		encoded_indicies = encoded_indicies.cuda()
	index = indices.view(indices.size()+torch.Size([1]))
	encoded_indicies = encoded_indicies.scatter_(1, index, 1)

	return encoded_indicies


def crop_lmark_center_mouth(img, lmark):
	def center_ab(a, b, c, d):
		return [(b[0]+a[0])//2, (b[1]+a[1])//2]

	lmark_2 = lmark[:, :2]
	c = center_ab(lmark_2[48], lmark_2[54], lmark_2[51], lmark_2[57])
	return c


def accuracy(output, target, topk=(1,)):
	# https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840/4
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
		pred = pred.t()
   
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(1.0 / batch_size))
		return res
