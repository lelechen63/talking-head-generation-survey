import os
import pdb
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision.models as models
from torchvision.models import vgg19

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def get_vgg19(opt, pretrained=True):
	# pdb.set_trace()
	net = vgg19(pretrained=pretrained)
	net = nn.Sequential(*list(net.features.children())[:-1])

	return net


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, opt, layers, num_classes=512):
		self.inplanes = 64
		self.opt = opt
		super(ResNet, self).__init__()
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(2)
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.bnfc = nn.BatchNorm1d(num_classes)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x).mean(-1).mean(-1)
		x = self.fc(x)
		x = self.bnfc(x)
		return x


def resnet34(options, pretrained=False):
	model = ResNet(BasicBlock, options, [3, 4, 6, 3])
	origin_dict = model.state_dict()
	if pretrained:
		pretrain_dict = model_zoo.load_url(model_urls['resnet34'])
		pretrain_dict = {k: v for k,
						 v in pretrain_dict.items() if k in origin_dict.keys() and 'fc' not in k}
		origin_dict.update(pretrain_dict)
		model.load_state_dict(origin_dict)

	return model


class D3Conv233(nn.Module):
	def __init__(self, opt):
		super(D3Conv233, self).__init__()
		self.features = nn.Sequential(OrderedDict([
			('conv0', nn.Conv3d(opt.in_channel, 64, kernel_size=(
				5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)),
			('norm0', nn.BatchNorm3d(64)),
			('relu0', nn.ReLU(inplace=True)),
			('pool0', nn.MaxPool3d(kernel_size=(1, 3, 3),
								   stride=(1, 2, 2), padding=(0, 1, 1))),
		]))

	def forward(self, x):
		x = self.features(x)

		return x


class D3Conv_Res34_233(nn.Module):
	def __init__(self, opt):
		super(D3Conv_Res34_233, self).__init__()
		self.opt = opt
		self.d3conv = D3Conv233(opt)
		self.res = resnet34(opt)

	def forward(self, feat):
		feat = feat.transpose(1, 2)
		feat = self.d3conv(feat)
		feat = feat.transpose(1, 2)
		b = feat.size(0)
		feat = feat.reshape([-1]+list(feat.size())[2:])
		feat = self.res(feat)
		feat = feat.reshape(b, -1, feat.size(-1))

		return feat
