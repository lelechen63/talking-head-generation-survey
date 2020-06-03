import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv1_1 = self.__conv(2, name='conv1_1', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv1_2 = self.__conv(2, name='conv1_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_1 = self.__conv(2, name='conv2_1', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_2 = self.__conv(2, name='conv2_2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_1 = self.__conv(2, name='conv3_1', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_2 = self.__conv(2, name='conv3_2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_3 = self.__conv(2, name='conv3_3', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_4 = self.__conv(2, name='conv3_4', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_1 = self.__conv(2, name='conv4_1', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_2 = self.__conv(2, name='conv4_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_3 = self.__conv(2, name='conv4_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_4 = self.__conv(2, name='conv4_4', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_1 = self.__conv(2, name='conv5_1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_2 = self.__conv(2, name='conv5_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_3 = self.__conv(2, name='conv5_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_4 = self.__conv(2, name='conv5_4', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.fc6_1 = self.__dense(name = 'fc6_1', in_features = 25088, out_features = 4096, bias = True)
        self.fc7_1 = self.__dense(name = 'fc7_1', in_features = 4096, out_features = 4096, bias = True)
        self.fc8_1 = self.__dense(name = 'fc8_1', in_features = 4096, out_features = 1000, bias = True)

    def forward(self, x):
        conv1_1_pad     = F.pad(x, (1, 1, 1, 1))
        conv1_1         = self.conv1_1(conv1_1_pad)
        relu1_1         = F.relu(conv1_1)
        conv1_2_pad     = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2         = self.conv1_2(conv1_2_pad)
        relu1_2         = F.relu(conv1_2)
        pool1_pad       = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad     = F.pad(pool1, (1, 1, 1, 1))
        conv2_1         = self.conv2_1(conv2_1_pad)
        relu2_1         = F.relu(conv2_1)
        conv2_2_pad     = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2         = self.conv2_2(conv2_2_pad)
        relu2_2         = F.relu(conv2_2)
        pool2_pad       = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2           = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad     = F.pad(pool2, (1, 1, 1, 1))
        conv3_1         = self.conv3_1(conv3_1_pad)
        relu3_1         = F.relu(conv3_1)
        conv3_2_pad     = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2         = self.conv3_2(conv3_2_pad)
        relu3_2         = F.relu(conv3_2)
        conv3_3_pad     = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3         = self.conv3_3(conv3_3_pad)
        relu3_3         = F.relu(conv3_3)
        conv3_4_pad     = F.pad(relu3_3, (1, 1, 1, 1))
        conv3_4         = self.conv3_4(conv3_4_pad)
        relu3_4         = F.relu(conv3_4)
        pool3_pad       = F.pad(relu3_4, (0, 1, 0, 1), value=float('-inf'))
        pool3           = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad     = F.pad(pool3, (1, 1, 1, 1))
        conv4_1         = self.conv4_1(conv4_1_pad)
        relu4_1         = F.relu(conv4_1)
        conv4_2_pad     = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2         = self.conv4_2(conv4_2_pad)
        relu4_2         = F.relu(conv4_2)
        conv4_3_pad     = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3         = self.conv4_3(conv4_3_pad)
        relu4_3         = F.relu(conv4_3)
        conv4_4_pad     = F.pad(relu4_3, (1, 1, 1, 1))
        conv4_4         = self.conv4_4(conv4_4_pad)
        relu4_4         = F.relu(conv4_4)
        pool4_pad       = F.pad(relu4_4, (0, 1, 0, 1), value=float('-inf'))
        pool4           = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad     = F.pad(pool4, (1, 1, 1, 1))
        conv5_1         = self.conv5_1(conv5_1_pad)
        relu5_1         = F.relu(conv5_1)
        conv5_2_pad     = F.pad(relu5_1, (1, 1, 1, 1))
        conv5_2         = self.conv5_2(conv5_2_pad)
        relu5_2         = F.relu(conv5_2)
        conv5_3_pad     = F.pad(relu5_2, (1, 1, 1, 1))
        conv5_3         = self.conv5_3(conv5_3_pad)
        relu5_3         = F.relu(conv5_3)
        conv5_4_pad     = F.pad(relu5_3, (1, 1, 1, 1))
        conv5_4         = self.conv5_4(conv5_4_pad)
        relu5_4         = F.relu(conv5_4)
        pool5_pad       = F.pad(relu5_4, (0, 1, 0, 1), value=float('-inf'))
        pool5           = F.max_pool2d(pool5_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        fc6_0           = pool5.view(pool5.size(0), -1)
        fc6_1           = self.fc6_1(fc6_0)
        relu6           = F.relu(fc6_1)
        drop6           = F.dropout(input = relu6, p = 0.5, training = self.training, inplace = True)
        fc7_0           = drop6.view(drop6.size(0), -1)
        fc7_1           = self.fc7_1(fc7_0)
        relu7           = F.relu(fc7_1)
        drop7           = F.dropout(input = relu7, p = 0.5, training = self.training, inplace = True)
        fc8_0           = drop7.view(drop7.size(0), -1)
        fc8_1           = self.fc8_1(fc8_0)
        prob            = F.softmax(fc8_1)
        return prob


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

